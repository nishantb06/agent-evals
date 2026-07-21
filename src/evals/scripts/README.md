# Batch therapist evals

Scripts to sample multi-turn windows from `therapist_pairs.csv`, run the
Ollive agent under several model profiles, and score every agent reply with
the four Gemini judges (hallucination, bias/harm, jailbreak, empathy), using
the therapist reply as a reference exemplar.

## Prerequisites

- Gateway running on `:8108` with `GEMINI_API_KEY` (judges + gemini agent) and
  `NVIDIA_API_KEY` (llama profiles).
- `therapist_pairs.csv` at the repo root (from `parquet_to_csv.py`).
- From this package: `cd src/evals && uv sync`.

## Commands

```bash
# Terminal 1 — gateway
cd src/gateway && uv run main.py

# Terminal 2 — sample + eval
cd src/evals
uv sync

# 1) Sample 10 conversations × 10 exchanges
uv run python scripts/sample_conversations.py \
  --input ../../therapist_pairs.csv \
  --output data/sampled_conversations.csv \
  --num-conversations 10 --exchanges 10 --seed 42

# 2) Smoke one model first (recommended)
uv run python scripts/run_batch_eval.py \
  --input data/sampled_conversations.csv \
  --models gemini \
  --out-dir results

# 3) Full A/B across profiles
uv run python scripts/run_batch_eval.py \
  --input data/sampled_conversations.csv \
  --models gemini,llama-3,llama-3-8b \
  --out-dir results

# 4) Re-print / recompute summary from a results CSV
uv run python scripts/summarize_report.py results/batch_eval_<timestamp>.csv
uv run python scripts/summarize_report.py results/batch_eval_<timestamp>.csv \
  --markdown results/batch_eval_<timestamp>_report.md
```

## Outputs

| File | Contents |
|------|----------|
| `data/sampled_conversations.csv` | `conversation_id,turn_index,source_row,query,therapist_response` |
| `results/batch_eval_<ts>.csv` | Per-turn agent reply + judge scores / rationales / violations |
| `results/batch_eval_<ts>_summary.csv` | Per model × criterion: `n`, `mean`, `variance` (sample) |

Default persona (marriage counselor) is baked into `run_batch_eval.py`; override
with `--persona "..."`.
