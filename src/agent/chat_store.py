"""Persistent chat transcripts for multi-turn conversations.

A chat is independent of a graph session:
  state/chats/<chat_id>/
    meta.json          # persona, channel, timestamps
    conversation.json  # ordered user/assistant turns linked to run_id

Each user message still runs as its own `run-*` graph session; this store
only holds the ordered dialogue that adapters (CLI, Telegram, …) share.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from persistence import _atomic_write

CHATS_ROOT = Path(__file__).parent / "state" / "chats"

# Allow adapter prefixes like cli:, tg:, discord: plus alphanumerics.
_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

# Caps for the prompt-facing history window. Full transcript stays on disk.
DEFAULT_MAX_TURNS = 12          # complete messages (user+assistant counted separately)
DEFAULT_MAX_CHARS = 6_000


class ChatIdError(ValueError):
    """Raised when a chat_id is empty, malformed, or escapes the chats root."""


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    run_id: str | None = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatMeta(BaseModel):
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    persona: str | None = None
    channel: str | None = None


def validate_chat_id(chat_id: str) -> str:
    if not chat_id or not _CHAT_ID_RE.match(chat_id):
        raise ChatIdError(f"invalid chat_id: {chat_id!r}")
    if ".." in chat_id or chat_id.startswith("/") or chat_id.startswith("\\"):
        raise ChatIdError(f"invalid chat_id: {chat_id!r}")
    return chat_id


class ChatStore:
    """One on-disk chat transcript under state/chats/<chat_id>/."""

    def __init__(self, chat_id: str, *, root: Path | None = None):
        self.chat_id = validate_chat_id(chat_id)
        self.root = Path(root) if root is not None else CHATS_ROOT
        self.dir = (self.root / self.chat_id).resolve()
        # Path-traversal guard after resolve.
        root_resolved = self.root.resolve()
        if self.dir != root_resolved and root_resolved not in self.dir.parents:
            raise ChatIdError(f"chat_id escapes chats root: {chat_id!r}")

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def conversation_path(self) -> Path:
        return self.dir / "conversation.json"

    def create_or_open(
        self,
        *,
        persona: str | None = None,
        channel: str | None = None,
    ) -> ChatMeta:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            meta = ChatMeta.model_validate_json(self.meta_path.read_text(encoding="utf-8"))
            dirty = False
            if persona is not None and persona != meta.persona:
                meta.persona = persona
                dirty = True
            if channel is not None and channel != meta.channel:
                meta.channel = channel
                dirty = True
            if dirty:
                meta.updated_at = datetime.now(timezone.utc)
                self._write_meta(meta)
            if not self.conversation_path.exists():
                _atomic_write(self.conversation_path, "[]")
            return meta

        meta = ChatMeta(persona=persona, channel=channel)
        self._write_meta(meta)
        _atomic_write(self.conversation_path, "[]")
        return meta

    def _write_meta(self, meta: ChatMeta) -> None:
        _atomic_write(
            self.meta_path,
            json.dumps(meta.model_dump(mode="json"), indent=2, ensure_ascii=False),
        )

    def read_meta(self) -> ChatMeta | None:
        if not self.meta_path.exists():
            return None
        return ChatMeta.model_validate_json(self.meta_path.read_text(encoding="utf-8"))

    def read_turns(self) -> list[ChatTurn]:
        if not self.conversation_path.exists():
            return []
        raw = json.loads(self.conversation_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [ChatTurn.model_validate(item) for item in raw]

    def append_turn(self, turn: ChatTurn) -> None:
        self.create_or_open()
        turns = self.read_turns()
        turns.append(turn)
        payload = [t.model_dump(mode="json") for t in turns]
        _atomic_write(
            self.conversation_path,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )
        meta = self.read_meta() or ChatMeta()
        meta.updated_at = datetime.now(timezone.utc)
        self._write_meta(meta)

    def recent_history(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_chars: int = DEFAULT_MAX_CHARS,
        exclude_trailing_user: bool = True,
    ) -> list[ChatTurn]:
        """Return a bounded window of prior turns for prompt injection.

        By default drops a trailing unfinished user turn so the message
        currently being answered is not duplicated inside CHAT HISTORY.
        """
        turns = self.read_turns()
        if exclude_trailing_user and turns and turns[-1].role == "user":
            turns = turns[:-1]

        # Prefer complete pairs from the end, then trim by char budget.
        window = turns[-max_turns:] if max_turns > 0 else list(turns)
        while window and sum(len(t.content) for t in window) > max_chars:
            window.pop(0)
        return window

    def history_dicts(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_chars: int = DEFAULT_MAX_CHARS,
        exclude_trailing_user: bool = True,
    ) -> list[dict]:
        """Shape compatible with memory.read(history=...)."""
        return [
            {"role": t.role, "content": t.content, "run_id": t.run_id}
            for t in self.recent_history(
                max_turns=max_turns,
                max_chars=max_chars,
                exclude_trailing_user=exclude_trailing_user,
            )
        ]

    def format_history(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_chars: int = DEFAULT_MAX_CHARS,
        exclude_trailing_user: bool = True,
    ) -> str:
        turns = self.recent_history(
            max_turns=max_turns,
            max_chars=max_chars,
            exclude_trailing_user=exclude_trailing_user,
        )
        if not turns:
            return ""
        lines: list[str] = []
        for t in turns:
            label = "User" if t.role == "user" else "Assistant"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)
