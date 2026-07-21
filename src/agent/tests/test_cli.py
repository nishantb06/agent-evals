"""Unit tests for flow.py CLI argv parsing (_parse_cli)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flow import CliParseError, _parse_cli


def test_persona_alone_is_repl() -> None:
    p = _parse_cli(["--persona", "You are a nutrition coach"])
    assert p.mode == "repl"
    assert p.chat_id is None
    assert p.persona == "You are a nutrition coach"


def test_chat_and_persona_either_order() -> None:
    a = _parse_cli(["--chat", "cli-xyz", "--persona", "be kind"])
    b = _parse_cli(["--persona", "be kind", "--chat", "cli-xyz"])
    assert a.mode == b.mode == "repl"
    assert a.chat_id == b.chat_id == "cli-xyz"
    assert a.persona == b.persona == "be kind"


def test_chat_alone_is_repl() -> None:
    p = _parse_cli(["--chat", "cli-abc"])
    assert p.mode == "repl"
    assert p.chat_id == "cli-abc"
    assert p.persona is None


def test_empty_argv_is_repl() -> None:
    p = _parse_cli([])
    assert p.mode == "repl"
    assert p.chat_id is None
    assert p.persona is None


def test_missing_persona_value_errors() -> None:
    with pytest.raises(CliParseError, match="--persona requires"):
        _parse_cli(["--persona"])
    with pytest.raises(CliParseError, match="--persona requires"):
        _parse_cli(["--persona", "--chat", "cli-x"])


def test_missing_chat_value_errors() -> None:
    with pytest.raises(CliParseError, match="--chat requires"):
        _parse_cli(["--chat"])


def test_oneshot_positional_query() -> None:
    p = _parse_cli(["hello", "world"])
    assert p.mode == "oneshot"
    assert p.query == "hello world"


def test_resume_recognized() -> None:
    p = _parse_cli(["--resume", "run-abc", "optional", "query"])
    assert p.mode == "resume"
    assert p.resume_sid == "run-abc"
    assert p.query == "optional query"


def test_resume_without_sid_errors() -> None:
    with pytest.raises(CliParseError, match="--resume requires"):
        _parse_cli(["--resume"])


def test_persona_plus_positional_rejected() -> None:
    with pytest.raises(CliParseError, match="positional query"):
        _parse_cli(["--persona", "coach", "what should I eat"])
