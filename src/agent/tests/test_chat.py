"""Unit tests for ChatStore + handle_turn (mocked Executor)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat import ChatTurnResult, handle_turn, new_chat_id
from chat_store import ChatIdError, ChatStore, ChatTurn, validate_chat_id
from skills import render_prompt


@pytest.fixture()
def chats_root(tmp_path: Path) -> Path:
    root = tmp_path / "chats"
    root.mkdir()
    return root


def test_validate_chat_id_accepts_adapter_prefixes() -> None:
    assert validate_chat_id("cli-abc12345") == "cli-abc12345"
    assert validate_chat_id("tg:12345") == "tg:12345"
    assert validate_chat_id("discord:channel-1") == "discord:channel-1"


def test_validate_chat_id_rejects_traversal() -> None:
    with pytest.raises(ChatIdError):
        validate_chat_id("../etc")
    with pytest.raises(ChatIdError):
        validate_chat_id("")
    with pytest.raises(ChatIdError):
        validate_chat_id("/abs/path")


def test_chat_store_roundtrip(chats_root: Path) -> None:
    store = ChatStore("cli-test01", root=chats_root)
    meta = store.create_or_open(persona="be kind", channel="cli")
    assert meta.persona == "be kind"
    assert meta.channel == "cli"

    store.append_turn(ChatTurn(role="user", content="hello", run_id="run-aaaa"))
    store.append_turn(ChatTurn(role="assistant", content="hi there", run_id="run-aaaa"))

    turns = store.read_turns()
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].content == "hi there"
    assert turns[1].run_id == "run-aaaa"

    store2 = ChatStore("cli-test01", root=chats_root)
    meta2 = store2.create_or_open()
    assert meta2.persona == "be kind"
    assert len(store2.read_turns()) == 2


def test_recent_history_excludes_trailing_user_and_caps(chats_root: Path) -> None:
    store = ChatStore("cli-hist01", root=chats_root)
    store.create_or_open()
    for i in range(5):
        store.append_turn(ChatTurn(role="user", content=f"u{i}", run_id=f"run-{i}"))
        store.append_turn(ChatTurn(role="assistant", content=f"a{i}", run_id=f"run-{i}"))
    store.append_turn(ChatTurn(role="user", content="pending", run_id="run-pending"))

    hist = store.recent_history(max_turns=4, exclude_trailing_user=True)
    assert all(t.content != "pending" for t in hist)
    assert len(hist) <= 4
    assert hist[-1].role == "assistant"

    tiny = store.recent_history(max_turns=20, max_chars=10, exclude_trailing_user=True)
    assert sum(len(t.content) for t in tiny) <= 10 or len(tiny) <= 1


def test_format_history_labels(chats_root: Path) -> None:
    store = ChatStore("cli-fmt01", root=chats_root)
    store.create_or_open()
    store.append_turn(ChatTurn(role="user", content="Q1", run_id="run-1"))
    store.append_turn(ChatTurn(role="assistant", content="A1", run_id="run-1"))
    text = store.format_history()
    assert "User: Q1" in text
    assert "Assistant: A1" in text


def test_new_chat_id_prefix() -> None:
    cid = new_chat_id("cli")
    assert cid.startswith("cli-")
    assert len(cid) > 4


def test_render_prompt_chat_context_planner_only() -> None:
    planner = SimpleNamespace(
        name="planner",
        prompt_template=lambda: "You are the Planner.",
    )
    researcher = SimpleNamespace(
        name="researcher",
        prompt_template=lambda: "You are the Researcher.",
    )
    resolved = [{"id": "USER_QUERY", "kind": "query", "value": "follow up"}]
    hist = "User: first\nAssistant: answer one"
    p = render_prompt(
        planner, "follow up", resolved,
        chat_context=hist, persona="be supportive",
    )
    assert "CHAT HISTORY (previous turns):" in p
    assert "User: first" in p
    assert "PERSONA:\nbe supportive" in p
    assert "USER_QUERY: follow up" in p

    r = render_prompt(
        researcher, "follow up", resolved,
        chat_context=hist, persona="be supportive",
    )
    assert "CHAT HISTORY" not in r
    assert "PERSONA" not in r


def test_render_prompt_no_history_matches_baseline() -> None:
    skill = SimpleNamespace(name="planner", prompt_template=lambda: "BASE")
    resolved = [{"id": "USER_QUERY", "kind": "query", "value": "hi"}]
    a = render_prompt(skill, "hi", resolved)
    b = render_prompt(skill, "hi", resolved, chat_context=None, persona=None)
    assert a == b


class _FakeExecutor:
    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, query, *, session_id=None, resume=False,
                  chat_history=None, chat_context=None, persona=None):
        self.calls.append({
            "query": query,
            "session_id": session_id,
            "chat_history": chat_history,
            "chat_context": chat_context,
            "persona": persona,
        })
        return f"echo:{query}"


def test_handle_turn_links_run_ids(chats_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import chat as chat_mod

    class RootedStore(ChatStore):
        def __init__(self, chat_id, *, root=None):
            super().__init__(chat_id, root=chats_root)

    monkeypatch.setattr(chat_mod, "ChatStore", RootedStore)

    fake = _FakeExecutor()
    result = asyncio.run(
        handle_turn("cli-ht01", "first question", executor=fake, channel="cli")
    )
    assert isinstance(result, ChatTurnResult)
    assert result.chat_id == "cli-ht01"
    assert result.run_id.startswith("run-")
    assert result.answer == "echo:first question"
    assert fake.calls[0]["chat_history"] is None
    assert fake.calls[0]["session_id"] == result.run_id

    result2 = asyncio.run(handle_turn("cli-ht01", "follow up", executor=fake))
    assert result2.run_id != result.run_id
    assert fake.calls[1]["chat_history"]
    assert fake.calls[1]["chat_history"][0]["content"] == "first question"
    assert "User: first question" in (fake.calls[1]["chat_context"] or "")

    store = ChatStore("cli-ht01", root=chats_root)
    turns = store.read_turns()
    assert len(turns) == 4
    assert turns[0].role == "user" and turns[0].run_id == result.run_id
    assert turns[1].role == "assistant" and turns[1].run_id == result.run_id
    assert turns[2].run_id == result2.run_id
    assert turns[3].run_id == result2.run_id


def test_handle_turn_rejects_empty() -> None:
    with pytest.raises(ValueError):
        asyncio.run(handle_turn("cli-empty", "   ", executor=_FakeExecutor()))


def test_handle_turn_failure_keeps_user_turn(
    chats_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chat as chat_mod

    class RootedStore(ChatStore):
        def __init__(self, chat_id, *, root=None):
            super().__init__(chat_id, root=chats_root)

    monkeypatch.setattr(chat_mod, "ChatStore", RootedStore)

    class Boom:
        async def run(self, *a, **k):
            raise RuntimeError("gateway down")

    with pytest.raises(RuntimeError):
        asyncio.run(handle_turn("cli-fail01", "will fail", executor=Boom()))

    turns = ChatStore("cli-fail01", root=chats_root).read_turns()
    assert len(turns) == 1
    assert turns[0].role == "user"
    assert turns[0].content == "will fail"
