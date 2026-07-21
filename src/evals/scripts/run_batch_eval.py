#!/usr/bin/env python3
"""Run multi-turn agent evals against sampled therapist conversations.

Independent (model × conversation) jobs run concurrently (--concurrency).
Turns within one conversation stay sequential (multi-turn history).

Writes / resumes:
  results/batch_eval_<timestamp>.csv          — per-turn detail
  results/batch_eval_<timestamp>_summary.csv  — mean / variance by model × criterion

  --resume PATH appends to an existing detail CSV and skips completed turns.

Usage (from src/evals, gateway on :8108):
  uv run python scripts/run_batch_eval.py \\
    --input data/sampled_conversations.csv \\
    --models gemini,llama-3,llama-3-8b \\
    --concurrency 6 \\
    --resume results/batch_eval_20260721T143624Z.csv
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
from typing import Any

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

RowKey = tuple[str, str, int]  # model, conversation_id, turn_index


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


def _empty_score_cols(criteria: tuple[str, ...]) -> dict:
    out: dict = {}
    for c in criteria:
        out[f"{c}_score"] = ""
        out[f"{c}_rationale"] = ""
        out[f"{c}_violations"] = "[]"
    out["hallucination_kb"] = "[]"
    return out


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


def _row_ok(row: dict) -> bool:
    return bool((row.get("agent_response") or "").strip()) and not (row.get("error") or "").strip()


def _row_key(row: dict) -> RowKey:
    return (
        row["model_profile"],
        row["conversation_id"],
        int(row["turn_index"]),
    )


def _load_resume_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dedupe_last_ok(rows: list[dict]) -> list[dict]:
    """Keep last successful row per (model, conv, turn); else last row."""
    best: dict[RowKey, dict] = {}
    order: list[RowKey] = []
    for row in rows:
        key = _row_key(row)
        if key not in best:
            order.append(key)
        prev = best.get(key)
        if prev is None or _row_ok(row) or not _row_ok(prev):
            best[key] = row
    return [best[k] for k in order]


def _write_summary(rows: list[dict], criteria: tuple[str, ...], path: Path) -> None:
    rows = _dedupe_last_ok(rows)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not _row_ok(row):
            continue
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
                    mean: Any = ""
                    var: Any = ""
                elif n == 1:
                    mean = round(vals[0], 4)
                    var = 0.0
                else:
                    mean = round(statistics.mean(vals), 4)
                    var = round(statistics.variance(vals), 4)
                w.writerow({
                    "model_profile": model,
                    "criterion": c,
                    "n": n,
                    "mean": mean,
                    "variance": var,
                })


def _progress_for_job(
    existing: list[dict],
    model: str,
    cid: str,
    n_turns: int,
) -> tuple[int, str | None, list[dict]]:
    """Return (start_turn, chat_id, conversation_prefix) for resume."""
    done_turns: dict[int, dict] = {}
    for row in existing:
        if row.get("model_profile") != model or row.get("conversation_id") != cid:
            continue
        if not _row_ok(row):
            continue
        ti = int(row["turn_index"])
        done_turns[ti] = row  # last ok wins

    start = 0
    while start < n_turns and start in done_turns:
        start += 1

    conversation: list[dict] = []
    chat_id: str | None = None
    for ti in range(start):
        row = done_turns[ti]
        conversation.append({"role": "user", "content": row.get("query") or ""})
        conversation.append({
            "role": "assistant",
            "content": row.get("agent_response") or "",
            "run_id": row.get("run_id") or "",
        })
        chat_id = (row.get("chat_id") or "").strip() or chat_id

    return start, chat_id, conversation


async def _run_one_conversation(
    *,
    model: str,
    cid: str,
    turns: list[dict],
    persona: str,
    kb_corpus: list,
    criteria: tuple[str, ...],
    handle_turn,
    judge_turn,
    sem: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    writer: csv.DictWriter,
    file_handle,
    detail_rows: list[dict],
    existing: list[dict],
    progress: dict,
) -> None:
    async with sem:
        n_turns = len(turns)
        start, chat_id, conversation = _progress_for_job(
            existing, model, cid, n_turns,
        )
        if start >= n_turns:
            print(f"  skip {model}/{cid} (already complete)", flush=True)
            return

        print(
            f"  start {model}/{cid} from turn {start}/{n_turns - 1}"
            + (f" chat={chat_id}" if chat_id else ""),
            flush=True,
        )

        async def run_from(start_turn: int, chat: str | None, conv: list[dict],
                           restart: bool) -> None:
            nonlocal chat_id
            chat_id = chat
            conversation.clear()
            conversation.extend(conv)

            if restart:
                # Drop in-memory rows for this job so summary prefers restart.
                kept = [
                    r for r in detail_rows
                    if not (
                        r.get("model_profile") == model
                        and r.get("conversation_id") == cid
                    )
                ]
                detail_rows.clear()
                detail_rows.extend(kept)

            for turn in turns[start_turn:]:
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
                    **_empty_score_cols(criteria),
                }

                if not query:
                    row["error"] = "empty_query"
                    async with write_lock:
                        writer.writerow(row)
                        file_handle.flush()
                        detail_rows.append(row)
                        progress["done"] += 1
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
                    row.update(_flatten_scores(scores, criteria))
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    row["error"] = err
                    print(f"    FAIL {model}/{cid} turn {turn_index}: {err}", flush=True)
                    async with write_lock:
                        writer.writerow(row)
                        file_handle.flush()
                        detail_rows.append(row)
                        progress["done"] += 1
                    if start_turn > 0 and turn_index == start_turn:
                        # Mid-resume failed on first continued turn → full restart.
                        print(
                            f"  restart {model}/{cid} from turn 0 after mid-resume failure",
                            flush=True,
                        )
                        await run_from(0, None, [], restart=True)
                        return
                    # Later failure: stop this conversation (partial).
                    return

                async with write_lock:
                    writer.writerow(row)
                    file_handle.flush()
                    detail_rows.append(row)
                    progress["done"] += 1
                    d, t = progress["done"], progress["total"]
                print(f"    ok {model}/{cid} turn {turn_index} ({d}/{t})", flush=True)

        await run_from(start, chat_id, conversation, restart=False)


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
    fieldnames = _result_fieldnames(CRITERIA)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if args.resume:
        resume_path = args.resume
        if not resume_path.is_file():
            print(f"--resume not found: {resume_path}", file=sys.stderr)
            return 1
        existing = _load_resume_csv(resume_path)
        detail_path = resume_path
        summary_path = resume_path.with_name(resume_path.stem + "_summary.csv")
        print(f"resuming → {detail_path} ({len(existing)} existing rows)", flush=True)
        mode = "a"
        write_header = False
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        detail_path = args.out_dir / f"batch_eval_{ts}.csv"
        summary_path = args.out_dir / f"batch_eval_{ts}_summary.csv"
        mode = "w"
        write_header = True

    detail_rows: list[dict] = list(existing)
    total_turns = sum(len(t) for t in conversations.values()) * len(models)
    # Approximate remaining for progress display.
    done_keys = {_row_key(r) for r in existing if _row_ok(r)}
    already = 0
    for model in models:
        for cid, turns in conversations.items():
            for turn in turns:
                key = (model, cid, int(turn["turn_index"]))
                if key in done_keys:
                    already += 1
    progress = {"done": already, "total": total_turns}

    sem = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()

    with detail_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        jobs = [
            _run_one_conversation(
                model=model,
                cid=cid,
                turns=turns,
                persona=persona,
                kb_corpus=kb_corpus,
                criteria=CRITERIA,
                handle_turn=handle_turn,
                judge_turn=judge_turn,
                sem=sem,
                write_lock=write_lock,
                writer=writer,
                file_handle=f,
                detail_rows=detail_rows,
                existing=existing,
                progress=progress,
            )
            for model in models
            for cid, turns in conversations.items()
        ]
        print(
            f"launching {len(jobs)} jobs "
            f"(concurrency={args.concurrency}, already_done≈{already}/{total_turns})",
            flush=True,
        )
        await asyncio.gather(*jobs)

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
    p.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Max parallel (model × conversation) jobs (default 6)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Append to an existing batch_eval_*.csv, skipping completed turns",
    )
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        print("Run sample_conversations.py first.", file=sys.stderr)
        return 1

    os.environ.setdefault(
        "LLM_GATEWAY_URL",
        os.getenv("LLM_GATEWAY_V8_URL", "http://localhost:8108"),
    )

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
