"""LLM-as-judge orchestration: four concurrent Gemini calls via the gateway."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from kb_retrieval import format_kb_section, load_kb, retrieve_chunks

_HERE = Path(__file__).resolve().parent
_PROMPTS = _HERE / "prompts"
_GATEWAY_DIR = (_HERE.parent / "gateway").resolve()

# Load gateway client without putting gateway/ on sys.path (schemas collision).
import importlib.util as _ilu

_client_path = _GATEWAY_DIR / "client.py"
_spec = _ilu.spec_from_file_location("ollive_gateway_client_evals", _client_path)
_mod = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
LLM = _mod.LLM

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "rationale": {"type": "string"},
        "violations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["score", "rationale", "violations"],
    "additionalProperties": False,
}

CRITERIA = ("hallucination", "bias_harm", "jailbreak", "empathy")

_PROMPT_FILES = {
    "hallucination": "hallucination.md",
    "bias_harm": "bias_harm.md",
    "jailbreak": "jailbreak.md",
    "empathy": "empathy.md",
}

_AGENT_TAGS = {
    "hallucination": "judge_hallucination",
    "bias_harm": "judge_bias",
    "jailbreak": "judge_jailbreak",
    "empathy": "judge_empathy",
}


def load_prompt(criterion: str) -> str:
    path = _PROMPTS / _PROMPT_FILES[criterion]
    return path.read_text(encoding="utf-8").strip()


def format_transcript(conversation: list[dict], *, mark_last_assistant: bool = True) -> str:
    """Render conversation as USER:/ASSISTANT: lines; mark last assistant turn."""
    lines: list[str] = []
    last_asst_idx = -1
    if mark_last_assistant:
        for i, turn in enumerate(conversation):
            if turn.get("role") == "assistant":
                last_asst_idx = i
    for i, turn in enumerate(conversation):
        role = (turn.get("role") or "user").upper()
        content = (turn.get("content") or "").strip()
        if i == last_asst_idx:
            lines.append(f"ASSISTANT (<<RESPONSE UNDER EVALUATION>>):\n{content}")
        else:
            label = "USER" if role == "USER" else "ASSISTANT"
            lines.append(f"{label}:\n{content}")
    return "\n\n".join(lines)


def build_user_payload(
    criterion: str,
    conversation: list[dict],
    *,
    kb_chunks: list[dict] | None = None,
) -> str:
    transcript = format_transcript(conversation)
    parts = [
        "CONVERSATION:",
        transcript,
    ]
    if criterion == "hallucination":
        parts += [
            "",
            "KNOWLEDGE BASE CHUNKS:",
            format_kb_section(kb_chunks or []),
        ]
    parts += [
        "",
        "Evaluate the RESPONSE UNDER EVALUATION per your instructions.",
        "Return JSON: {\"score\": <1-10>, \"rationale\": \"...\", \"violations\": []}",
    ]
    return "\n".join(parts)


def _clamp_score(raw: Any) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, n))


def _parse_judge_response(resp: dict) -> dict:
    parsed = resp.get("parsed")
    if isinstance(parsed, dict) and "score" in parsed:
        return {
            "score": _clamp_score(parsed.get("score")),
            "rationale": str(parsed.get("rationale") or "").strip(),
            "violations": list(parsed.get("violations") or []),
            "error": None,
        }
    # Fallback: try loose JSON in text
    text = (resp.get("text") or "").strip()
    if text:
        import json
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(text[start:end + 1])
                return {
                    "score": _clamp_score(obj.get("score")),
                    "rationale": str(obj.get("rationale") or "").strip(),
                    "violations": list(obj.get("violations") or []),
                    "error": None,
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {
        "score": None,
        "rationale": text[:500] if text else "judge returned no parseable score",
        "violations": [],
        "error": "parse_failed",
    }


def _sync_judge_one(
    llm: Any,
    criterion: str,
    conversation: list[dict],
    kb_chunks: list[dict] | None,
) -> dict:
    system = load_prompt(criterion)
    user = build_user_payload(criterion, conversation, kb_chunks=kb_chunks)
    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            provider="gemini",
            temperature=1.0,
            max_tokens=1024,
            agent=_AGENT_TAGS[criterion],
            response_format={
                "type": "json_schema",
                "schema": JUDGE_SCHEMA,
                "name": f"judge_{criterion}",
                "strict": True,
            },
        )
        out = _parse_judge_response(resp)
    except Exception as e:
        out = {
            "score": None,
            "rationale": f"{type(e).__name__}: {e}",
            "violations": [],
            "error": "gateway_error",
        }
    if criterion == "hallucination":
        out["kb_chunks_used"] = [
            {
                "source": c["source"],
                "index": c["index"],
                "score": c["score"],
                "preview": c["text"][:200],
            }
            for c in (kb_chunks or [])
        ]
    return out


async def judge_turn(
    conversation: list[dict],
    *,
    kb_corpus: list | None = None,
    llm: Any | None = None,
) -> dict:
    """Judge the last assistant turn with four concurrent Gemini calls."""
    if not conversation:
        raise ValueError("conversation is empty")
    if conversation[-1].get("role") != "assistant":
        raise ValueError("conversation must end with an assistant turn")

    # Build query from last user + assistant for KB retrieval.
    last_user = ""
    for turn in reversed(conversation[:-1]):
        if turn.get("role") == "user":
            last_user = turn.get("content") or ""
            break
    last_asst = conversation[-1].get("content") or ""
    kb_query = f"{last_user}\n{last_asst}"
    kb_chunks = retrieve_chunks(kb_query, kb=kb_corpus)

    client = llm or LLM()
    loop = asyncio.get_running_loop()

    async def one(criterion: str) -> tuple[str, dict]:
        chunks = kb_chunks if criterion == "hallucination" else None
        result = await loop.run_in_executor(
            None,
            lambda c=criterion, ch=chunks: _sync_judge_one(client, c, conversation, ch),
        )
        return criterion, result

    gathered = await asyncio.gather(*(one(c) for c in CRITERIA))
    return {name: payload for name, payload in gathered}


async def judge_conversation(
    conversation: list[dict],
    *,
    kb_corpus: list | None = None,
    llm: Any | None = None,
) -> dict:
    """Judge every assistant turn with prefix context. Returns per-turn + averages."""
    turns_out: list[dict] = []
    sums = {c: 0.0 for c in CRITERIA}
    counts = {c: 0 for c in CRITERIA}

    for i, turn in enumerate(conversation):
        if turn.get("role") != "assistant":
            continue
        prefix = conversation[: i + 1]
        scores = await judge_turn(prefix, kb_corpus=kb_corpus, llm=llm)
        entry = {
            "turn_index": i,
            "run_id": turn.get("run_id"),
            "content_preview": (turn.get("content") or "")[:160],
            "scores": scores,
        }
        turns_out.append(entry)
        for c in CRITERIA:
            s = scores.get(c, {}).get("score")
            if isinstance(s, int):
                sums[c] += s
                counts[c] += 1

    averages = {
        c: (round(sums[c] / counts[c], 2) if counts[c] else None)
        for c in CRITERIA
    }
    return {
        "turns": turns_out,
        "averages": averages,
        "turn_count": len(turns_out),
    }
