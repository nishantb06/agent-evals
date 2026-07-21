# Session Graph Viewer

Local UI for inspecting agent execution graphs. Supports **chat-first** navigation (pick a chat, then a linked graph session) and a **Memory / FAISS hits** panel.

The older standalone file [`../graph-viewer.html`](../graph-viewer.html) (manual file pickers) is left in place; prefer this folder.

## Run

```bash
cd graph-viewer

# defaults: ../agent/state/sessions and ../agent/state/chats
python server.py

# optional paths / port
python server.py \
  --sessions-dir ../agent/state/sessions \
  --chats-dir ../agent/state/chats \
  --port 8765
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/).

Python 3.10+ stdlib only — no `uv sync` / extra packages.

## UI

- **Chat dropdown** — `<chat-id>: <first user message>` plus **Standalone sessions** (graph runs not linked from any chat transcript).
- **Session dropdown** — for a chat, only that chat’s `run_id`s in conversation order; for standalone, unlinked sessions newest-first.
- **Query banner** — chat id + turn index, and the session `query.txt`.
- **Memory / FAISS hits** — kind, source, descriptor, expandable chunk text; badge shows whether hits came from `memory_hits.json` or were parsed from `prompt_sent`.
- **Retriever chunks** — when a retriever node ran, its `output.chunks` appear under the memory panel.
- **Hover / click** — same graph node inspection as before.

Edges: recorded NetworkX edges, plus inferred `n:` input deps and spawn edges from `result.successors`.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/chats` | `[{ id, preview, turn_count, run_ids, … }]` newest first |
| GET | `/api/chats/<id>` | `{ id, meta, conversation, runs: [{ run_id, query, exists }] }` |
| GET | `/api/sessions` | all graph sessions |
| GET | `/api/sessions/standalone` | sessions not referenced by any chat |
| GET | `/api/sessions/<id>` | `{ id, query, graph, nodes, memory_hits, memory_hits_source, retriever_chunks }` |

Read-only. Ids are validated to block path traversal.

## What is / isn’t stored

| Data | Available in viewer? |
|------|----------------------|
| Chat ↔ session links via `run_id` | Yes (`state/chats/*/conversation.json`) |
| Memory hits for a run | Yes — prefer `sessions/<id>/memory_hits.json` (written by new agent runs); else parse `MEMORY HITS` from planner `prompt_sent` |
| Retrieved chunk text | Yes (from hits file / prompt / retriever output) |
| FAISS similarity scores | **No** — never persisted |
| Full untruncated chunks on old runs | Best-effort — prompt previews are capped (~2000 chars) |

New graph runs (via `flow.py` / `handle_turn`) write `memory_hits.json` next to `graph.json` after `memory.read`.
