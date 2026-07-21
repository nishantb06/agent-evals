#!/usr/bin/env python3
"""Extract <query>/<response> spans from a parquet file into a CSV.

Expects columns `Query` and `Response`. For each row:
  - client input  = text between <query> and </query> in Query
  - therapist reply = text between <response> and </response> in Response

Usage:
  uv run --with pandas --with pyarrow python parquet_to_csv.py \\
      train-00000-of-00001.parquet -o therapist_pairs.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_QUERY_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL | re.IGNORECASE)
_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)


def extract(text: object, pattern: re.Pattern[str]) -> str | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    m = pattern.search(str(text))
    if not m:
        return None
    return m.group(1).strip()


def convert(parquet_path: Path, csv_path: Path, *, drop_incomplete: bool) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path)
    for col in ("Query", "Response"):
        if col not in df.columns:
            raise SystemExit(
                f"missing column {col!r}; found: {list(df.columns)}"
            )

    rows: list[dict[str, str]] = []
    skipped = 0
    for _, row in df.iterrows():
        query = extract(row["Query"], _QUERY_RE)
        response = extract(row["Response"], _RESPONSE_RE)
        if query is None or response is None:
            skipped += 1
            if drop_incomplete:
                continue
            query = query or ""
            response = response or ""
        rows.append({"query": query, "response": response})

    out = pd.DataFrame(rows, columns=["query", "response"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    return len(out), skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", type=Path, help="Input .parquet path")
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .csv path (default: <parquet_stem>.csv)",
    )
    p.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep rows missing a tag (empty string) instead of dropping them",
    )
    args = p.parse_args(argv)

    parquet_path = args.parquet
    if not parquet_path.is_file():
        print(f"file not found: {parquet_path}", file=sys.stderr)
        return 1

    csv_path = args.output or parquet_path.with_suffix(".csv")
    n, skipped = convert(
        parquet_path,
        csv_path,
        drop_incomplete=not args.keep_incomplete,
    )
    print(f"wrote {n} rows → {csv_path}  (skipped incomplete: {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
