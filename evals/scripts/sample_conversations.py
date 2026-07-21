#!/usr/bin/env python3
"""Sample contiguous multi-turn conversation windows from therapist_pairs.csv.

Each conversation is EXCHANGES contiguous rows (default 10). Samples
NUM_CONVERSATIONS non-overlapping windows at random.

Output columns:
  conversation_id, turn_index, source_row, query, therapist_response

Usage (from evals/):
  uv run python scripts/sample_conversations.py \\
    --input ../therapist_pairs.csv \\
    --output data/sampled_conversations.csv \\
    --num-conversations 10 --exchanges 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def sample_windows(
    n_rows: int,
    *,
    num_conversations: int,
    exchanges: int,
    seed: int,
) -> list[int]:
    """Return sorted list of start indices for non-overlapping windows."""
    if exchanges < 1:
        raise ValueError("exchanges must be >= 1")
    if num_conversations < 1:
        raise ValueError("num_conversations must be >= 1")
    max_start = n_rows - exchanges
    if max_start < 0:
        raise ValueError(
            f"need at least {exchanges} rows; file has {n_rows}"
        )
    # All possible non-overlapping placements via packing random starts.
    rng = random.Random(seed)
    # Greedy: shuffle candidate starts, take first that don't overlap.
    candidates = list(range(0, max_start + 1))
    rng.shuffle(candidates)
    chosen: list[int] = []
    occupied: list[tuple[int, int]] = []
    for start in candidates:
        end = start + exchanges
        if any(not (end <= a or start >= b) for a, b in occupied):
            continue
        chosen.append(start)
        occupied.append((start, end))
        if len(chosen) >= num_conversations:
            break
    if len(chosen) < num_conversations:
        raise ValueError(
            f"could only place {len(chosen)} non-overlapping windows of "
            f"{exchanges} in {n_rows} rows; request fewer conversations "
            f"or shorter exchanges"
        )
    return sorted(chosen)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "therapist_pairs.csv",
        help="Source CSV with query,response columns",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sampled_conversations.csv",
    )
    p.add_argument("--num-conversations", type=int, default=10)
    p.add_argument("--exchanges", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "query" not in reader.fieldnames or "response" not in reader.fieldnames:
            print(
                f"expected columns query,response; got {reader.fieldnames}",
                file=sys.stderr,
            )
            return 1
        rows = list(reader)

    starts = sample_windows(
        len(rows),
        num_conversations=args.num_conversations,
        exchanges=args.exchanges,
        seed=args.seed,
    )

    out_rows: list[dict] = []
    for ci, start in enumerate(starts):
        cid = f"conv-{ci:02d}"
        for t in range(args.exchanges):
            src = start + t
            r = rows[src]
            out_rows.append({
                "conversation_id": cid,
                "turn_index": t,
                "source_row": src,
                "query": (r.get("query") or "").strip(),
                "therapist_response": (r.get("response") or "").strip(),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_id",
                "turn_index",
                "source_row",
                "query",
                "therapist_response",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(
        f"wrote {len(out_rows)} rows "
        f"({args.num_conversations} × {args.exchanges}) → {args.output}"
    )
    print(f"starts (source_row): {starts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
