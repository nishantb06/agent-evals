#!/usr/bin/env python3
"""Run multi-turn agent evals against sampled therapist conversations.

For each model profile × conversation:
  - open a fresh chat with the marriage-counselor persona
  - send each client query via handle_turn (agent history, not therapist)
  - score the agent reply with judge_turn(..., reference_response=therapist)

Writes:
  results/batch_eval_<timestamp>.csv          — per-turn detail
  results/batch_eval_<timestamp>_summary.csv  — mean / variance by model × criterion

Usage (from src/evals, gateway on :8108):
  uv run python scripts/run_batch_eval.py \\
    --input data/sampled_conversations.csv \\
    --models gemini,llama-3,llama-3-8b \\
    --out-dir results
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALS_DIR = HERE.parent
SRC = EVALS_DIR.parent
AGENT_DIR = (SRC / "agent").resolve()
GATEWAY_DIR = (SRC / "gateway").resolve()

DEFAULT_PERSONA = (
    "You are an expert marriage counselor sitting with a couple. "
    "Help them work through their relationship challenges. Be warm, "
    "inquisitive, and non-judgmental; ask clarifying open-ended questions "
    "before offering guidance. Stay non-diagnostic and encourage professional "
    "care when issues exceed wellness support."
)


def _bootstrap_paths() -> None:
    sys.path.insert(0, str(AGENT_DIR))
    sys.path.insert(0, str(EVALS_DIR))
    for candidate in (SRC / ".env", GATEWAY_DIR / ".env", EVALS_DIR / ".env"):
        if candidate.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate)
                break
            except OSError:
                continue


def _load_sampled(path: Path) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"conversation_id", "turn_index", "query", "therapist_response"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"sampled CSV missing columns {required}; got {reader.fieldnames}"
            )
        by_conv: dict[str, list[dict]] = defaultdict(list)
        for row in reader:
            by_conv[row["conversation_id"]].append(row)
    for cid, turns in by_conv.items():
        turns.sort(key=lambda r: int(r["turn_index"]))
    return dict(by_conv)


def _result_fieldnames(criteria: tuple[str, ...]) -> list[str]:
    cols = [
        "model_profile",
        "conversation_id",
        "turn_index",
        "query",
        "therapist_response",
        "agent_response",
        "chat_id",
        "run_id",
        "error",
    ]
    for c in criteria:
        cols += [f"{c}_score", f"{c}_rationale", f"{c}_violations"]
    cols.append("hallucination_kb")
    return cols


def _flatten_scores(scores: dict, criteria: tuple[str, ...]) -> dict:
    out: dict = {}
    for c in criteria:
        s = scores.get(c) or {}
        out[f"{c}_score"] = s.get("score")
        out[f"{c}_rationale"] = s.get("rationale") or ""
        viol = s.get("violations") or []
        out[f"{c}_violations"] = json.dumps(viol, ensure_ascii=False)
    kb = (scores.get("hallucination") or {}).get("kb_chunks_used") or []
    out["hallucination_kb"] = json.dumps(kb, ensure_ascii=False)
    return out


def _write_summary(rows: list[dict], criteria: tuple[str, ...], path: Path) -> None:
    # model → criterion → list of scores
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = row["model_profile"]
        for c in criteria:
            raw = row.get(f"{c}_score")
            if raw is None or raw == "":
                continue
            try:
                buckets[model][c].append(float(raw))
            except (TypeError, ValueError):
                continue

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model_profile", "criterion", "n", "mean", "variance"],
        )
        w.writeheader()
        for model in sorted(buckets):
            for c in criteria:
                vals = buckets[model][c]
                n = len(vals)
                if n == 0:
                    mean = ""
                    var = ""
                elif n == 1:
                    mean = round(vals[0], 4)
                    var = 0.0
                else:
                    mean = round(statistics.mean(vals), 4)
                    var = round(statistics.variance(vals), 4)  # sample, ddof=1
                w.writerow({
                    "model_profile": model,
                    "criterion": c,
                    "n": n,
                    "mean": mean,
                    "variance": var,
                })


async def _run(args: argparse.Namespace) -> int:
    _bootstrap_paths()

    from agent_model import resolve  # noqa: E402
    from chat import handle_turn  # noqa: E402
    from judge import CRITERIA, judge_turn  # noqa: E402
    from kb_retrieval import load_kb  # noqa: E402

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        try:
            resolve(m)
        except Exception as e:
            print(f"unknown model profile {m!r}: {e}", file=sys.stderr)
            return 1

    conversations = _load_sampled(args.input)
    kb_corpus = load_kb()
    persona = args.persona or DEFAULT_PERSONA

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / f"batch_eval_{ts}.csv"
    summary_path = args.out_dir / f"batch_eval_{ts}_summary.csv"
    fieldnames = _result_fieldnames(CRITERIA)

    detail_rows: list[dict] = []
    total = sum(len(t) for t in conversations.values()) * len(models)
    done = 0

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model in models:
            print(f"\n=== model={model} ===", flush=True)
            for cid, turns in conversations.items():
                chat_id = None
                conversation: list[dict] = []
                print(f"  conversation {cid} ({len(turns)} turns)", flush=True)
                for turn in turns:
                    query = (turn.get("query") or "").strip()
                    therapist = (turn.get("therapist_response") or "").strip()
                    turn_index = int(turn["turn_index"])
                    row: dict = {
                        "model_profile": model,
                        "conversation_id": cid,
                        "turn_index": turn_index,
                        "query": query,
                        "therapist_response": therapist,
                        "agent_response": "",
                        "chat_id": chat_id or "",
                        "run_id": "",
                        "error": "",
                    }
                    for c in CRITERIA:
                        row[f"{c}_score"] = ""
                        row[f"{c}_rationale"] = ""
                        row[f"{c}_violations"] = "[]"
                    row["hallucination_kb"] = "[]"

                    if not query:
                        row["error"] = "empty_query"
                        writer.writerow(row)
                        detail_rows.append(row)
                        done += 1
                        continue

                    try:
                        result = await handle_turn(
                            chat_id,
                            query,
                            persona=persona if chat_id is None else None,
                            model_profile=model,
                            channel="batch_eval",
                        )
                        chat_id = result.chat_id
                        conversation.append({"role": "user", "content": query})
                        conversation.append({
                            "role": "assistant",
                            "content": result.answer,
                            "run_id": result.run_id,
                        })
                        row["agent_response"] = result.answer
                        row["chat_id"] = result.chat_id
                        row["run_id"] = result.run_id

                        scores = await judge_turn(
                            conversation,
                            kb_corpus=kb_corpus,
                            reference_response=therapist,
                        )
                        row.update(_flatten_scores(scores, CRITERIA))
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {e}"
                        print(
                            f"    turn {turn_index} FAILED: {row['error']}",
                            flush=True,
                        )

                    writer.writerow(row)
                    f.flush()
                    detail_rows.append(row)
                    done += 1
                    print(
                        f"    turn {turn_index} done ({done}/{total})",
                        flush=True,
                    )

    _write_summary(detail_rows, CRITERIA, summary_path)
    print(f"\nwrote detail  → {detail_path}")
    print(f"wrote summary → {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=EVALS_DIR / "data" / "sampled_conversations.csv",
    )
    p.add_argument(
        "--models",
        default="gemini,llama-3,llama-3-8b",
        help="Comma-separated model profiles",
    )
    p.add_argument("--persona", default=None, help="Override default counselor persona")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=EVALS_DIR / "results",
    )
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        print("Run sample_conversations.py first.", file=sys.stderr)
        return 1

    # Ensure gateway URL default for agent client.
    os.environ.setdefault(
        "LLM_GATEWAY_URL",
        os.getenv("LLM_GATEWAY_V8_URL", "http://localhost:8108"),
    )

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
