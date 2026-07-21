"""Evals platform FastAPI server — live chat + LLM-as-judge UI."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
SRC = HERE.parent                          # …/src
REPO = SRC.parent                          # repo root
AGENT_DIR = (SRC / "agent").resolve()
GATEWAY_DIR = (SRC / "gateway").resolve()
STATIC_DIR = HERE / "static"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Env: prefer src/.env then gateway/.env (ignore permission errors in sandbox)
for candidate in (SRC / ".env", GATEWAY_DIR / ".env", HERE / ".env"):
    if candidate.exists():
        try:
            load_dotenv(candidate)
            break
        except OSError:
            continue

sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(HERE))

from judge import judge_conversation, judge_turn  # noqa: E402
from kb_retrieval import load_kb  # noqa: E402

GATEWAY_URL = (
    os.getenv("LLM_GATEWAY_URL")
    or os.getenv("LLM_GATEWAY_V8_URL")
    or "http://localhost:8108"
).rstrip("/")

# Warm KB corpus once at import.
KB_CORPUS = load_kb()

app = FastAPI(title="Ollive Evals", version="0.1.0")


class ChatRequest(BaseModel):
    message: str
    chat_id: str | None = None
    persona: str | None = None
    model_profile: str = "gemini"


class ChatResponse(BaseModel):
    chat_id: str
    run_id: str
    answer: str


class JudgeRequest(BaseModel):
    conversation: list[dict] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    role: str
    content: str
    run_id: str | None = None
    ts: str | None = None


@app.get("/api/health")
async def health():
    gateway_ok = False
    gateway_error = None
    try:
        r = httpx.get(f"{GATEWAY_URL}/v1/routers", timeout=3.0)
        gateway_ok = r.status_code == 200
        if not gateway_ok:
            gateway_error = f"status {r.status_code}"
    except Exception as e:
        gateway_error = f"{type(e).__name__}: {e}"
    return {
        "ok": True,
        "gateway_url": GATEWAY_URL,
        "gateway_ok": gateway_ok,
        "gateway_error": gateway_error,
        "kb_chunks": len(KB_CORPUS),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    text = (req.message or "").strip()
    if not text:
        raise HTTPException(400, "message must be non-empty")
    try:
        from chat import handle_turn
        result = await handle_turn(
            req.chat_id,
            text,
            channel="evals",
            persona=req.persona,
            model_profile=req.model_profile,
        )
    except Exception as e:
        raise HTTPException(502, f"agent error: {type(e).__name__}: {e}") from e
    return ChatResponse(
        chat_id=result.chat_id,
        run_id=result.run_id,
        answer=result.answer,
    )


@app.post("/api/judge")
async def api_judge(req: JudgeRequest):
    conv = req.conversation or []
    if not conv:
        raise HTTPException(400, "conversation is empty")
    if conv[-1].get("role") != "assistant":
        raise HTTPException(400, "conversation must end with an assistant turn")
    try:
        scores = await judge_turn(conv, kb_corpus=KB_CORPUS)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"judge error: {type(e).__name__}: {e}") from e
    return scores


@app.post("/api/eval-file")
async def api_eval_file(turns: list[ConversationTurn]):
    if not turns:
        raise HTTPException(400, "conversation array is empty")
    conversation = [t.model_dump() for t in turns]
    asst = sum(1 for t in conversation if t.get("role") == "assistant")
    if asst == 0:
        raise HTTPException(400, "no assistant turns to evaluate")
    try:
        result = await judge_conversation(conversation, kb_corpus=KB_CORPUS)
    except Exception as e:
        raise HTTPException(502, f"eval error: {type(e).__name__}: {e}") from e

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval-{stamp}.json"
    payload = {
        "saved_at": stamp,
        "source": "upload",
        "conversation": conversation,
        **result,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    result["saved_path"] = str(out_path.name)
    return result


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("EVALS_PORT", "8901"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
