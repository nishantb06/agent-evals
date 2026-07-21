# Ollive — Multi-Agent Graph Orchestrator

A multi-agent growing-graph orchestrator. The graph itself is the agent
loop: each node is a typed skill (Planner, Researcher, Distiller, Critic,
Formatter, Coder, …), edges carry the predecessor's `AgentResult`, and the
runtime executes ready nodes in parallel via `asyncio.gather`.

---

## Layout

```
src/
├── README.md          ← you are here
├── .env.example       ← copy to .env, fill in keys you have
│
├── agent/             ← the agent. Run from here.
│   ├── flow.py        ← orchestrator (Graph + Executor + CLI)
│   ├── chat.py        ← multi-turn handle_turn() wrapper (adapters call this)
│   ├── chat_store.py  ← persistent chat transcripts under state/chats/
│   ├── skills.py      ← skill registry, prompt rendering, run_skill
│   ├── recovery.py    ← failure classification + critic-fail splice
│   ├── persistence.py ← session writes (graph.json + per-node JSON)
│   ├── mcp_runner.py  ← multi-turn tool-use loop wrapper
│   ├── sandbox.py     ← subprocess Python runner (usability boundary; NOT security)
│   ├── replay.py      ← stdin-driven trace viewer
│   ├── schemas.py     ← AgentResult, NodeSpec, NodeState, MemoryItem, …
│   ├── agent_config.yaml  ← skills catalogue
│   ├── prompts/       ← one .md per skill
│   ├── tests/
│   ├── mcp_server.py  ← MCP tools: web_search, fetch_url, search_knowledge, …
│   ├── memory.py / vector_index.py / artifacts.py
│   ├── perception.py / decision.py / action.py
│   └── sandbox/papers/  ← sample corpus for indexed-knowledge queries
│
├── gateway/           ← LLM Gateway (FastAPI). Runs on :8108.
│   ├── main.py
│   ├── client.py      ← the SDK agent/gateway.py imports from
│   ├── providers.py / router.py / embedders.py / db.py / cache.py
│   ├── agent_routing.yaml  ← agent → preferred provider mapping
│   ├── pyproject.toml
│   └── run.sh
│
└── graph-viewer/      ← local UI for inspecting execution graphs
```

---

## Quickstart

You need: Python 3.11+, [uv](https://docs.astral.sh/uv/), Ollama
(`brew install ollama` then `ollama pull nomic-embed-text`), and at least
one provider API key from `.env.example`.

```bash
# 1. Secrets
cp .env.example .env
$EDITOR .env                  # add the keys you have

# 2. Install
cd gateway && uv sync && cd ..
cd agent   && uv sync && cd ..

# 3. Start the gateway (one terminal)
cd gateway && uv run main.py
# (or: ./run.sh)
# It boots on http://localhost:8108; /v1/routers should answer.

# 4. Run the agent (another terminal)
cd agent
uv run python flow.py "hello"          # one-shot
# or multi-turn REPL:
uv run python flow.py                  # new chat
uv run python flow.py --chat cli-….    # reopen a chat by id
uv run python flow.py --persona "You are a nutrition coach..."
uv run python flow.py --chat cli-…. --persona "You are a nutrition coach..."
```

A successful first one-shot run prints two node lines (planner, formatter)
and a greeting. Graph runs land in `agent/state/sessions/<sid>/`. Walk one
with:

```bash
uv run python replay.py <sid>
```

### Multi-turn chat

Chat is a thin layer over graph sessions: one user message → one fresh
`run-*` session → one Formatter answer. Transcripts live in
`agent/state/chats/<chat_id>/` (`meta.json` + `conversation.json`).
Optional `--persona` is stored on the chat in `meta.json` and injected
into Planner and Formatter prompts on every turn.

```bash
cd agent
uv run python flow.py
# you> According to my docs, what is farm-to-table?
# agent> …
# [run run-……  chat cli-……]
# you> summarise that in one sentence
# agent> …
# you> /quit
```

```bash
uv run python flow.py --persona "You are a practical nutrition coach..."
uv run python flow.py --chat cli-xyz --persona "You are a practical nutrition coach..."
```

Omitting `--persona` when reopening a chat keeps the persona already
saved in that chat's `meta.json`.

REPL commands: `/help`, `/chat` (print chat id), `/quit`.

`--resume <run-id>` is **graph crash-resume**, not chat follow-up.

### Adapter contract

Telegram / Discord / a custom frontend should call only
`chat.handle_turn` — no graph logic in the transport:

```python
from chat import handle_turn

result = await handle_turn(
    chat_id="tg:12345",          # or discord:… / cli:…
    message=inbound_text,
    channel="telegram",
    persona=None,                # optional chat-level persona
)
await send_reply(result.answer)
# result.run_id → state/sessions/<run_id>/ for replay
```

---

## Architecture

The Planner reads the user query and emits a small DAG of skill nodes
to run. Each ready node fires through the gateway in parallel with its
ready siblings. When a skill's yaml entry has `internal_successors`,
the orchestrator appends those automatically — that's how **Coder →
SandboxExecutor** chains without the Planner having to ask for it.

Critic nodes get auto-inserted on edges out of skills tagged
`critic: true` in `agent_config.yaml` (currently Distiller). A
verdict=fail from a Critic splices a recovery Planner into the graph,
capped at one re-plan per branch.

Failure handling is in `recovery.py`. Transient gateway errors don't
re-plan (the gateway already retries); validation errors don't re-plan
(it's a prompt bug); upstream-failures do. `tests/test_recovery.py`
pins the classifier against the actual gateway error strings.

---

## When things go wrong

| symptom | first place to look |
|---|---|
| `[gateway] launching … failed to start within 45s` | `cd gateway && uv run main.py` in another terminal; read its stderr. Probably a missing API key or port :8108 already taken. |
| `httpx.HTTPStatusError: '503 Service Unavailable'` | All worker providers in cooldown / unconfigured. Add another key to `.env` or wait a minute. |
| coder ran but `sandbox_executor` reports `no code in upstream coder output` | The Coder prompt isn't emitting the JSON shape the orchestrator expects (`{"code": "...", "rationale": "..."}`). |
| The final answer is short / wrong | Run `replay.py <sid>` and inspect what each node actually saw (the `prompt_sent` field captures the exact bytes sent to the gateway). |
