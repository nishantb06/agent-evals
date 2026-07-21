"""Chat-session wrapper over per-message graph runs.

Adapters (CLI, Telegram, HTTP, …) call `handle_turn` only. Each user
message becomes one fresh `run-*` Executor.run; the chat transcript links
those runs in order.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from chat_store import ChatStore, ChatTurn

# Per-chat locks so concurrent adapter deliveries cannot interleave writes.
_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


async def _lock_for(chat_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        lock = _LOCKS.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            _LOCKS[chat_id] = lock
        return lock


@dataclass
class ChatTurnResult:
    answer: str
    chat_id: str
    run_id: str


def new_chat_id(prefix: str = "cli") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def handle_turn(
    chat_id: str | None,
    message: str,
    *,
    persona: str | None = None,
    channel: str | None = None,
    executor=None,
) -> ChatTurnResult:
    """Run one user message as a new graph session and append both turns.

    `chat_id` may be None — a fresh cli-* id is generated. Prior transcript
    (excluding the message being appended) is passed into Executor as
    chat context for Planner/Formatter prompts and memory keyword fallback.
    """
    text = (message or "").strip()
    if not text:
        raise ValueError("message must be non-empty")

    cid = chat_id or new_chat_id("cli")
    lock = await _lock_for(cid)
    async with lock:
        store = ChatStore(cid)
        store.create_or_open(persona=persona, channel=channel)

        history_dicts = store.history_dicts(exclude_trailing_user=False) or None
        history_text = store.format_history(exclude_trailing_user=False) or None
        meta = store.read_meta()
        effective_persona = persona if persona is not None else (meta.persona if meta else None)

        run_id = f"run-{uuid.uuid4().hex[:8]}"
        store.append_turn(ChatTurn(role="user", content=text, run_id=run_id))

        # Lazy import keeps chat.py free of gateway side-effects at import time.
        if executor is None:
            from flow import Executor
            executor = Executor()

        try:
            answer = await executor.run(
                text,
                session_id=run_id,
                chat_history=history_dicts,
                chat_context=history_text,
                persona=effective_persona,
            )
        except Exception:
            # Leave the user turn on disk (with run_id) so operators can see
            # the failed attempt; do not invent an assistant reply.
            raise

        answer = (answer or "").strip()
        store.append_turn(ChatTurn(role="assistant", content=answer, run_id=run_id))
        return ChatTurnResult(answer=answer, chat_id=cid, run_id=run_id)
