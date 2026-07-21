#!/usr/bin/env python3
"""Recompute mean / variance summary from a batch_eval results CSV.

Usage:
  uv run python scripts/summarize_report.py results/batch_eval_<timestamp>.csv
  uv run python scripts/summarize_report.py results/batch_eval_<timestamp>.csv -o out.md
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CRITERIA = ("hallucination", "bias_harm", "jailbreak", "empathy")


def summarize(path: Path) -> list[dict]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model_profile") or ""
            for c in CRITERIA:
                raw = row.get(f"{c}_score")
                if raw is None or raw == "":
                    continue
                try:
                    buckets[model][c].append(float(raw))
                except (TypeError, ValueError):
                    continue

    rows: list[dict] = []
    for model in sorted(buckets):
        for c in CRITERIA:
            vals = buckets[model][c]
            n = len(vals)
            if n == 0:
                mean, var = None, None
            elif n == 1:
                mean, var = vals[0], 0.0
            else:
                mean = statistics.mean(vals)
                var = statistics.variance(vals)
            rows.append({
                "model_profile": model,
                "criterion": c,
                "n": n,
                "mean": None if mean is None else round(mean, 4),
                "variance": None if var is None else round(var, 4),
            })
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| model | criterion | n | mean | variance |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_profile']} | {r['criterion']} | {r['n']} | "
            f"{r['mean'] if r['mean'] is not None else '—'} | "
            f"{r['variance'] if r['variance'] is not None else '—'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_csv", type=Path, help="batch_eval_*.csv detail file")
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Write summary CSV here (default: <stem>_summary_recomputed.csv)",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write a markdown table to this path",
    )
    args = p.parse_args(argv)

    if not args.results_csv.is_file():
        print(f"not found: {args.results_csv}", file=sys.stderr)
        return 1

    rows = summarize(args.results_csv)
    out = args.output or args.results_csv.with_name(
        args.results_csv.stem + "_summary_recomputed.csv"
    )
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["model_profile", "criterion", "n", "mean", "variance"]
        )
        w.writeheader()
        w.writerows(rows)

    md = to_markdown(rows)
    print(md)
    if args.markdown:
        args.markdown.write_text(md, encoding="utf-8")
        print(f"wrote markdown → {args.markdown}")
    print(f"wrote summary → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
