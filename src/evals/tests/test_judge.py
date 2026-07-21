"""Tests for judge prompt rendering, score parsing, and eval-file with mocks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import judge
from judge import (
    CRITERIA,
    _parse_judge_response,
    build_user_payload,
    format_transcript,
    judge_conversation,
    judge_turn,
)


def test_format_transcript_marks_last_assistant() -> None:
    conv = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "final"},
    ]
    text = format_transcript(conv)
    assert "<<RESPONSE UNDER EVALUATION>>" in text
    assert text.count("RESPONSE UNDER EVALUATION") == 1
    assert "final" in text
    assert text.index("hello") < text.index("RESPONSE UNDER EVALUATION")


def test_build_user_payload_kb_only_for_hallucination() -> None:
    conv = [
        {"role": "user", "content": "supplements?"},
        {"role": "assistant", "content": "try ashwagandha"},
    ]
    chunks = [{"source": "08.md", "index": 0, "text": "Ashwagandha for stress", "score": 2}]
    h = build_user_payload("hallucination", conv, kb_chunks=chunks)
    b = build_user_payload("bias_harm", conv, kb_chunks=chunks)
    assert "KNOWLEDGE BASE CHUNKS" in h
    assert "Ashwagandha for stress" in h
    assert "KNOWLEDGE BASE CHUNKS" not in b


def test_parse_judge_response_from_parsed() -> None:
    out = _parse_judge_response({
        "parsed": {"score": 9, "rationale": "grounded", "violations": []},
        "text": "",
    })
    assert out["score"] == 9
    assert out["rationale"] == "grounded"
    assert out["error"] is None


def test_parse_judge_response_clamps_and_fallback_json() -> None:
    out = _parse_judge_response({
        "parsed": None,
        "text": 'Here you go:\n{"score": 99, "rationale": "x", "violations": ["a"]}\n',
    })
    assert out["score"] == 10
    assert out["violations"] == ["a"]


@pytest.mark.asyncio
async def test_judge_turn_three_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_sync(llm, criterion, conversation, kb_chunks):
        calls.append(criterion)
        return {
            "score": 8,
            "rationale": criterion,
            "violations": [],
            "error": None,
            **({"kb_chunks_used": []} if criterion == "hallucination" else {}),
        }

    monkeypatch.setattr(judge, "_sync_judge_one", fake_sync)
    monkeypatch.setattr(judge, "retrieve_chunks", lambda *a, **k: [])

    conv = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = await judge_turn(conv, kb_corpus=[], llm=MagicMock())
    assert set(result) == set(CRITERIA)
    assert set(calls) == set(CRITERIA)
    assert result["hallucination"]["score"] == 8


@pytest.mark.asyncio
async def test_judge_conversation_averages(monkeypatch: pytest.MonkeyPatch) -> None:
    scores_seq = iter([
        {"hallucination": {"score": 6}, "bias_harm": {"score": 8}, "jailbreak": {"score": 10}},
        {"hallucination": {"score": 8}, "bias_harm": {"score": 8}, "jailbreak": {"score": 10}},
    ])

    async def fake_turn(conversation, *, kb_corpus=None, llm=None):
        return next(scores_seq)

    monkeypatch.setattr(judge, "judge_turn", fake_turn)

    conv = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "run_id": "r1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2", "run_id": "r2"},
    ]
    out = await judge_conversation(conv, kb_corpus=[], llm=MagicMock())
    assert out["turn_count"] == 2
    assert out["averages"]["hallucination"] == 7.0
    assert out["averages"]["jailbreak"] == 10.0


def test_eval_file_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import server as server_mod

    async def fake_judge_conversation(conversation, *, kb_corpus=None, llm=None):
        return {
            "turns": [{
                "turn_index": 1,
                "run_id": "r1",
                "content_preview": "hello",
                "scores": {
                    "hallucination": {"score": 9, "rationale": "ok", "violations": []},
                    "bias_harm": {"score": 9, "rationale": "ok", "violations": []},
                    "jailbreak": {"score": 10, "rationale": "ok", "violations": []},
                },
            }],
            "averages": {"hallucination": 9.0, "bias_harm": 9.0, "jailbreak": 10.0},
            "turn_count": 1,
        }

    monkeypatch.setattr(server_mod, "judge_conversation", fake_judge_conversation)
    monkeypatch.setattr(server_mod, "RESULTS_DIR", tmp_path)

    client = TestClient(server_mod.app)
    payload = [
        {"role": "user", "content": "hi", "run_id": "r1"},
        {"role": "assistant", "content": "hello", "run_id": "r1"},
    ]
    res = client.post("/api/eval-file", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["turn_count"] == 1
    assert data["averages"]["jailbreak"] == 10.0
    assert data["saved_path"]
    saved = list(tmp_path.glob("eval-*.json"))
    assert len(saved) == 1
    disk = json.loads(saved[0].read_text())
    assert disk["turn_count"] == 1


def test_health_endpoint() -> None:
    import server as server_mod
    client = TestClient(server_mod.app)
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "kb_chunks" in data
    assert data["kb_chunks"] > 0


def test_chat_endpoint_passes_model_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    import server as server_mod
    from types import SimpleNamespace

    seen: dict = {}

    async def fake_handle_turn(chat_id, message, *, persona=None, channel=None,
                               model_profile=None, executor=None):
        seen["model_profile"] = model_profile
        seen["message"] = message
        return SimpleNamespace(
            chat_id=chat_id or "cli-test",
            run_id="run-test",
            answer="ok",
        )

    # Endpoint does `from chat import handle_turn` — patch the module attribute.
    sys.path.insert(0, str(server_mod.AGENT_DIR))
    import chat as chat_mod
    monkeypatch.setattr(chat_mod, "handle_turn", fake_handle_turn)

    client = TestClient(server_mod.app)
    res = client.post("/api/chat", json={
        "message": "hello",
        "model_profile": "llama-3",
    })
    assert res.status_code == 200
    assert seen["model_profile"] == "llama-3"
    assert res.json()["answer"] == "ok"
