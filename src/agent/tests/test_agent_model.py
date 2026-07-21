"""Tests for agent_model profiles and CLI --model parsing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_model import (
    UnknownModelProfile,
    get_chat_kwargs,
    resolve,
    set_profile,
)
from flow import CliParseError, _parse_cli


def test_resolve_gemini_default() -> None:
    p = resolve(None)
    assert p.name == "gemini"
    assert p.provider == "gemini"
    assert p.model is None


def test_resolve_llama_aliases() -> None:
    for alias in ("llama-3", "llama3", "llama", "Llama-3"):
        p = resolve(alias)
        assert p.name == "llama-3"
        assert p.provider == "nvidia"
        assert p.model == "meta/llama-3.1-70b-instruct"


def test_resolve_unknown() -> None:
    with pytest.raises(UnknownModelProfile):
        resolve("gpt-4")


def test_get_chat_kwargs_gemini_omits_model() -> None:
    set_profile("gemini")
    kw = get_chat_kwargs()
    assert kw == {"provider": "gemini"}


def test_get_chat_kwargs_llama() -> None:
    set_profile("llama-3")
    kw = get_chat_kwargs()
    assert kw == {
        "provider": "nvidia",
        "model": "meta/llama-3.1-70b-instruct",
    }


def test_cli_model_alone_is_repl() -> None:
    p = _parse_cli(["--model", "llama-3"])
    assert p.mode == "repl"
    assert p.model_profile == "llama-3"


def test_cli_model_with_chat_and_persona() -> None:
    p = _parse_cli([
        "--model", "llama-3",
        "--chat", "cli-xyz",
        "--persona", "be kind",
    ])
    assert p.mode == "repl"
    assert p.chat_id == "cli-xyz"
    assert p.persona == "be kind"
    assert p.model_profile == "llama-3"


def test_cli_model_oneshot() -> None:
    p = _parse_cli(["--model", "gemini", "hello", "world"])
    assert p.mode == "oneshot"
    assert p.query == "hello world"
    assert p.model_profile == "gemini"


def test_cli_unknown_model_errors() -> None:
    with pytest.raises(CliParseError, match="unknown model profile"):
        _parse_cli(["--model", "claude"])


def test_cli_resume_with_model_after() -> None:
    p = _parse_cli(["--resume", "run-abc", "--model", "llama-3"])
    assert p.mode == "resume"
    assert p.resume_sid == "run-abc"
    assert p.model_profile == "llama-3"


def test_run_skill_passes_llama_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """When profile is llama-3, LLM().chat must get nvidia + explicit model."""
    set_profile("llama-3")

    captured: dict = {}

    class FakeLLM:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"text": '{"final_answer": "ok"}', "provider": "nvidia", "cost": 0}

    monkeypatch.setattr("skills.LLM", FakeLLM)
    monkeypatch.setattr("skills.render_prompt", lambda *a, **k: "PROMPT")
    monkeypatch.setattr("skills.resolve_inputs", lambda *a, **k: {})
    monkeypatch.setattr(
        "skills.parse_skill_json",
        lambda t: {"final_answer": "ok"},
    )

    from schemas import AgentResult
    from skills import Skill, run_skill
    import asyncio

    skill = Skill("formatter", {
        "prompt": "prompts/formatter.md",
        "tools_allowed": [],
        "temperature": 0.3,
        "max_tokens": 200,
    })

    result, _prompt = asyncio.run(
        run_skill(
            skill, "n:1", {"n:1": {"inputs": [], "skill": "formatter"}},
            "run-test", "hi", None,
        )
    )
    assert isinstance(result, AgentResult)
    assert captured.get("provider") == "nvidia"
    assert captured.get("model") == "meta/llama-3.1-70b-instruct"
