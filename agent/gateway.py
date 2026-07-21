"""Bridge to the LLM gateway.

The gateway adds: (1) every `/v1/chat` accepts an optional `agent: str`
tag the gateway logs and uses for cost-by-agent rollups and provider
pinning; (2) a `/v1/chat/batch` endpoint that runs N chat requests
concurrently with bounded parallelism — what the DAG-style orchestrator
hits when firing a ready batch; (3) one retry on 5xx / timeout with
`retries` surfaced in the response.

Auto-starts the gateway on port 8108 if it is not already up, then
re-exports the `LLM` client and a module-level `embed()` helper.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx

# Sibling layout: agent/ and gateway/ at the repo root. Override with
# OLLIVE_GATEWAY_DIR if you move things.
import os as _os
GATEWAY_DIR = Path(
    _os.environ.get("OLLIVE_GATEWAY_DIR")
    or (Path(__file__).resolve().parent.parent / "gateway")
).resolve()
GATEWAY_URL = "http://localhost:8108"


def _is_up() -> bool:
    try:
        httpx.get(f"{GATEWAY_URL}/v1/routers", timeout=2.0)
        return True
    except Exception:
        return False


def ensure_gateway() -> None:
    """Start the gateway if it is not already running. Idempotent."""
    if _is_up():
        return
    if not GATEWAY_DIR.exists():
        raise RuntimeError(
            f"Gateway directory not found at {GATEWAY_DIR}. "
            "Ensure gateway/ is present at the repo root before running the agent."
        )
    print(f"[gateway] launching from {GATEWAY_DIR}")
    subprocess.Popen(
        ["uv", "run", "main.py"],
        cwd=str(GATEWAY_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(45):
        time.sleep(1)
        if _is_up():
            print(f"[gateway] up on {GATEWAY_URL}")
            return
    raise RuntimeError(f"Gateway failed to start within 45s. Check {GATEWAY_DIR}")


# Load the gateway client without polluting sys.path. The gateway dir has its
# own `schemas.py`, which would shadow ours if we put it on the path.
import importlib.util as _importlib_util

_client_path = GATEWAY_DIR / "client.py"
if _client_path.exists():
    _spec = _importlib_util.spec_from_file_location("ollive_gateway_client", _client_path)
    _mod = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LLM = _mod.LLM
else:
    LLM = None  # importers should call ensure_gateway() first


def embed(text: str, task_type: str = "retrieval_document") -> dict:
    """Compute an embedding for `text` via the gateway's embed endpoint.

    Returns the full response dict: `{embedding, dim, model, provider,
    latency_ms, ...}`. The chosen embedding model is fixed at the gateway
    level. Changing it invalidates every FAISS index built against the old
    vectors, so callers should treat the model as a project-level constant.
    """
    ensure_gateway()
    if LLM is None:
        raise RuntimeError(
            "Gateway client unavailable. Confirm gateway/client.py exists."
        )
    return LLM().embed(text, task_type=task_type)


__all__ = ["ensure_gateway", "LLM", "GATEWAY_URL", "GATEWAY_DIR", "embed"]
