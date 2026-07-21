# Ollive Evals Platform

LLM-as-judge evals for the Ollive agent. Chat with the agent in real time
(or upload a saved `conversation.json`) and score every assistant turn on:

1. **Hallucination / groundedness** (with KB fuzzy retrieval)
2. **Bias / harmful outputs**
3. **Jailbreak resistance**

All three judges call **Gemini** through the LLM gateway (`provider=gemini`).

## Prerequisites

- Gateway running on `:8108` with keys in `src/.env`:
  - `GEMINI_API_KEY` — required for judges **and** for `--model gemini` / UI Gemini
  - `NVIDIA_API_KEY` — required for Live Chat with **Llama-3 70B** or **8B (NVIDIA)**
- Run sync **from this directory** (`src/evals/`), not the repo root:
  ```bash
  cd src/evals
  uv sync
  ```
- KB present at `src/agent/sandbox/kb/` (chunked at startup)

`uv sync` here also installs the agent runtime deps (`networkx`, `faiss-cpu`,
`mcp`, …) so live chat can import `handle_turn`.

## Run

```bash
cd src/evals
uv sync
uv run server.py
# open http://localhost:8901
```

Optional: `EVALS_PORT=8901`, `LLM_GATEWAY_URL=http://localhost:8108`.

## Live Chat

1. Pick **agent model**: Gemini (frontier), Llama-3 70B, or Llama-3 8B (NVIDIA).
   Locked for the chat after the first message; use **New chat** to switch.
2. Optionally set a persona in the sidebar.
3. Send messages — each reply is judged with three concurrent **Gemini** calls
   (judges always use Gemini, regardless of agent model).
4. Click **H / B / J** chips on an assistant bubble for rationale, violations,
   and (for H) which KB chunks were used.

## Upload Eval

Upload a file like `src/agent/state/chats/<id>/conversation.json`:

```json
[
  {"role": "user", "content": "...", "run_id": "...", "ts": "..."},
  {"role": "assistant", "content": "...", "run_id": "...", "ts": "..."}
]
```

**Run eval** judges every assistant turn with full prefix context, shows
per-turn scores + averages, and saves `src/evals/results/eval-<timestamp>.json`.

## API

| Method | Path | Body |
|--------|------|------|
| GET | `/api/health` | — |
| POST | `/api/chat` | `{message, chat_id?, persona?, model_profile?}` (`gemini` \| `llama-3` \| `llama-3-8b`) |
| POST | `/api/judge` | `{conversation: [...]}` (must end with assistant) |
| POST | `/api/eval-file` | conversation.json array |

## Tests

```bash
cd src/evals
uv run pytest -q
```
