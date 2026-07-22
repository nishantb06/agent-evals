#!/usr/bin/env python3
"""Plot merged batch-eval summary CSVs into comparison infographics.

Usage (from evals/):
  uv run --with matplotlib python scripts/plot_eval_summaries.py \\
    --inputs \\
      results/batch_eval_20260721T143624Z_summary_recomputed.csv \\
      results/batch_eval_20260721T145905Z_summary_recomputed.csv \\
      results/batch_eval_20260721T150042Z_summary_recomputed.csv \\
    --out-dir ../images/report
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

CRITERIA = ("hallucination", "bias_harm", "jailbreak", "empathy")
CRITERION_LABELS = {
    "hallucination": "Hallucination",
    "bias_harm": "Bias / Harm",
    "jailbreak": "Jailbreak",
    "empathy": "Empathy",
}
MODEL_ORDER = ("gemini", "llama-3", "llama-3-8b")
MODEL_LABELS = {
    "gemini": "Gemini (frontier)",
    "llama-3": "Llama-3 70B",
    "llama-3-8b": "Llama-3 8B",
}
COLORS = {
    "gemini": "#4C8BF5",
    "llama-3": "#34A853",
    "llama-3-8b": "#FBBC04",
}


def _load(paths: list[Path]) -> dict[tuple[str, str], dict]:
    """(model, criterion) -> {n, mean, variance}."""
    out: dict[tuple[str, str], dict] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                model = (row.get("model_profile") or "").strip()
                crit = (row.get("criterion") or "").strip()
                if not model or crit not in CRITERIA:
                    continue
                try:
                    n = int(float(row["n"])) if row.get("n") not in (None, "") else 0
                except ValueError:
                    n = 0
                mean = None
                var = None
                if row.get("mean") not in (None, ""):
                    try:
                        mean = float(row["mean"])
                    except ValueError:
                        mean = None
                if row.get("variance") not in (None, ""):
                    try:
                        var = float(row["variance"])
                    except ValueError:
                        var = None
                out[(model, crit)] = {"n": n, "mean": mean, "variance": var}
    return out


def _models_present(data: dict) -> list[str]:
    found = {m for m, _ in data}
    return [m for m in MODEL_ORDER if m in found] or sorted(found)


def _grouped_bars(
    data: dict,
    *,
    value_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    annotate_n: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = _models_present(data)
    criteria = list(CRITERIA)
    x = np.arange(len(criteria))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140)
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0f1419")

    for i, model in enumerate(models):
        vals = []
        ns = []
        for c in criteria:
            cell = data.get((model, c), {})
            v = cell.get(value_key)
            vals.append(float(v) if v is not None else 0.0)
            ns.append(int(cell.get("n") or 0))
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.92,
            label=MODEL_LABELS.get(model, model),
            color=COLORS.get(model, "#888"),
            edgecolor="#1a2330",
            linewidth=0.5,
        )
        if annotate_n:
            for bar, n in zip(bars, ns):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.08,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#c8d0d8",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([CRITERION_LABELS[c] for c in criteria], color="#e8eef4")
    ax.set_ylabel(ylabel, color="#e8eef4")
    ax.set_title(title, color="#e8eef4", fontsize=13, pad=12)
    ax.tick_params(colors="#c8d0d8")
    for spine in ax.spines.values():
        spine.set_color("#2a3544")
    ax.yaxis.grid(True, color="#2a3544", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, labelcolor="#e8eef4")
    for t in leg.get_texts():
        t.set_color("#e8eef4")
    if value_key == "mean":
        ax.set_ylim(0, 10.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _sample_sizes(data: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = _models_present(data)
    criteria = list(CRITERIA)
    x = np.arange(len(criteria))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0f1419")

    for i, model in enumerate(models):
        ns = [int(data.get((model, c), {}).get("n") or 0) for c in criteria]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            ns,
            width * 0.92,
            label=MODEL_LABELS.get(model, model),
            color=COLORS.get(model, "#888"),
            edgecolor="#1a2330",
            linewidth=0.5,
        )
        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(n),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#c8d0d8",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CRITERION_LABELS[c] for c in criteria], color="#e8eef4")
    ax.set_ylabel("Scored turns (n)", color="#e8eef4")
    ax.set_title("Sample sizes — do not over-interpret small-n Llama bars", color="#e8eef4", fontsize=13, pad=12)
    ax.tick_params(colors="#c8d0d8")
    for spine in ax.spines.values():
        spine.set_color("#2a3544")
    ax.yaxis.grid(True, color="#2a3544", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor="#e8eef4")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more *_summary_recomputed.csv paths",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "images" / "report",
    )
    args = p.parse_args(argv)

    missing = [str(x) for x in args.inputs if not x.is_file()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")

    data = _load(args.inputs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _grouped_bars(
        data,
        value_key="mean",
        title="Mean judge scores by criterion (1–10)",
        ylabel="Mean score",
        out_path=args.out_dir / "mean_by_criterion.png",
        annotate_n=True,
    )
    _grouped_bars(
        data,
        value_key="variance",
        title="Score variance by criterion (sample variance)",
        ylabel="Variance",
        out_path=args.out_dir / "variance_by_criterion.png",
        annotate_n=False,
    )
    _sample_sizes(data, args.out_dir / "sample_sizes.png")

    print(f"wrote charts → {args.out_dir}")
    for name in ("mean_by_criterion.png", "variance_by_criterion.png", "sample_sizes.png"):
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
