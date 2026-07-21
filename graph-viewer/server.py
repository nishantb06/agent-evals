#!/usr/bin/env python3
"""Read-only HTTP server for the session / chat graph viewer.

Serves static files from this directory and exposes:

  GET /api/sessions
  GET /api/sessions/<id>
  GET /api/chats
  GET /api/chats/<id>

Usage:
  python server.py
  python server.py --sessions-dir ../agent/state/sessions \\
                   --chats-dir ../agent/state/chats --port 8765
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from memory_parse import extract_memory_hits_from_nodes, extract_retriever_chunks

HERE = Path(__file__).resolve().parent
DEFAULT_SESSIONS = (HERE / ".." / "agent" / "state" / "sessions").resolve()
DEFAULT_CHATS = (HERE / ".." / "agent" / "state" / "chats").resolve()

# Session ids look like run-a8609b26; chat ids may include adapter prefixes.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

SESSIONS_DIR: Path = DEFAULT_SESSIONS
CHATS_DIR: Path = DEFAULT_CHATS


def _safe_session_dir(session_id: str) -> Path | None:
    if not _SESSION_ID_RE.match(session_id):
        return None
    root = SESSIONS_DIR.resolve()
    target = (root / session_id).resolve()
    if root not in target.parents and target != root:
        return None
    if not target.is_dir():
        return None
    return target


def _safe_chat_dir(chat_id: str) -> Path | None:
    if not _CHAT_ID_RE.match(chat_id) or ".." in chat_id:
        return None
    root = CHATS_DIR.resolve()
    target = (root / chat_id).resolve()
    if root not in target.parents and target != root:
        return None
    if not target.is_dir():
        return None
    return target


def _read_session_query(session_id: str) -> str:
    d = _safe_session_dir(session_id)
    if d is None:
        return ""
    qp = d / "query.txt"
    return qp.read_text(encoding="utf-8").strip() if qp.exists() else ""


def _session_mtime(session_id: str) -> float:
    d = _safe_session_dir(session_id)
    if d is None:
        return 0.0
    return d.stat().st_mtime


def list_sessions() -> list[dict]:
    if not SESSIONS_DIR.is_dir():
        return []
    out: list[tuple[float, dict]] = []
    for child in SESSIONS_DIR.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "graph.json").exists():
            continue
        query_path = child / "query.txt"
        query = query_path.read_text(encoding="utf-8") if query_path.exists() else ""
        mtime = child.stat().st_mtime
        out.append((mtime, {"id": child.name, "query": query.strip()}))
    out.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in out]


def _ordered_run_ids(conversation: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        rid = turn.get("run_id")
        if isinstance(rid, str) and rid and rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def _all_chat_run_ids() -> set[str]:
    ids: set[str] = set()
    if not CHATS_DIR.is_dir():
        return ids
    for child in CHATS_DIR.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        conv_path = child / "conversation.json"
        if not conv_path.exists():
            continue
        try:
            conv = json.loads(conv_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(conv, list):
            ids.update(_ordered_run_ids(conv))
    return ids


def list_chats() -> list[dict]:
    if not CHATS_DIR.is_dir():
        return []
    out: list[tuple[float, dict]] = []
    for child in CHATS_DIR.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        conv_path = child / "conversation.json"
        meta_path = child / "meta.json"
        if not conv_path.exists():
            continue
        try:
            conv = json.loads(conv_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(conv, list):
            conv = []
        preview = ""
        for turn in conv:
            if isinstance(turn, dict) and turn.get("role") == "user":
                preview = str(turn.get("content") or "")
                break
        updated = child.stat().st_mtime
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ts = meta.get("updated_at") or meta.get("created_at")
                if isinstance(ts, str):
                    updated = max(updated, meta_path.stat().st_mtime)
            except (json.JSONDecodeError, OSError):
                pass
        run_ids = _ordered_run_ids(conv)
        out.append((
            updated,
            {
                "id": child.name,
                "preview": preview,
                "turn_count": len(conv),
                "updated_at": updated,
                "run_ids": run_ids,
            },
        ))
    out.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in out]


def load_chat(chat_id: str) -> dict | None:
    chat_dir = _safe_chat_dir(chat_id)
    if chat_dir is None:
        return None
    conv_path = chat_dir / "conversation.json"
    meta_path = chat_dir / "meta.json"
    if not conv_path.exists():
        return None
    try:
        conversation = json.loads(conv_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(conversation, list):
        conversation = []
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    run_ids = _ordered_run_ids(conversation)
    runs = []
    for rid in run_ids:
        runs.append({
            "run_id": rid,
            "query": _read_session_query(rid),
            "mtime": _session_mtime(rid),
            "exists": _safe_session_dir(rid) is not None,
        })
    return {
        "id": chat_id,
        "meta": meta,
        "conversation": conversation,
        "runs": runs,
        "run_ids": run_ids,
    }


def _load_memory_hits_file(session_dir: Path) -> list | None:
    path = session_dir / "memory_hits.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, list) else None


def load_session(session_id: str) -> dict | None:
    session_dir = _safe_session_dir(session_id)
    if session_dir is None:
        return None

    query_path = session_dir / "query.txt"
    query = query_path.read_text(encoding="utf-8").strip() if query_path.exists() else ""

    graph_path = session_dir / "graph.json"
    if not graph_path.exists():
        return None
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    nodes: dict[str, dict] = {}
    nodes_dir = session_dir / "nodes"
    if nodes_dir.is_dir():
        for path in sorted(nodes_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            nid = payload.get("node_id") or payload.get("id")
            if isinstance(nid, str) and nid:
                nodes[nid] = payload

    # Merge node-file fields onto graph nodes for a single client-side view.
    for gnode in graph.get("nodes") or []:
        nid = gnode.get("id")
        extra = nodes.get(nid) if isinstance(nid, str) else None
        if not extra:
            continue
        for key in ("prompt_sent", "started_at", "completed_at", "retries"):
            if key in extra and key not in gnode:
                gnode[key] = extra[key]
            elif key in extra and gnode.get(key) in (None, "", 0) and extra.get(key) not in (None, ""):
                gnode[key] = extra[key]
        if extra.get("result") and not gnode.get("result"):
            gnode["result"] = extra["result"]
        if extra.get("status") and not gnode.get("status"):
            gnode["status"] = extra["status"]

    file_hits = _load_memory_hits_file(session_dir)
    if file_hits is not None:
        memory_hits = file_hits
        memory_hits_source = "memory_hits.json"
    else:
        memory_hits = extract_memory_hits_from_nodes(nodes)
        memory_hits_source = "prompt_sent" if memory_hits else "none"

    retriever_chunks = extract_retriever_chunks(nodes)

    return {
        "id": session_id,
        "query": query,
        "graph": graph,
        "nodes": nodes,
        "memory_hits": memory_hits,
        "memory_hits_source": memory_hits_source,
        "retriever_chunks": retriever_chunks,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "OlliveGraphViewer/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        body = json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _not_found(self, msg: str = "not found") -> None:
        self._json(404, {"error": msg})

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(self.path.split("?", 1)[0])

        if path == "/api/sessions":
            self._json(200, list_sessions())
            return

        if path == "/api/sessions/standalone":
            linked = _all_chat_run_ids()
            standalone = [s for s in list_sessions() if s["id"] not in linked]
            self._json(200, standalone)
            return

        if path.startswith("/api/sessions/"):
            session_id = path[len("/api/sessions/") :].strip("/")
            if not session_id or "/" in session_id:
                self._not_found("invalid session id")
                return
            payload = load_session(session_id)
            if payload is None:
                self._not_found(f"session '{session_id}' not found")
                return
            self._json(200, payload)
            return

        if path == "/api/chats":
            self._json(200, list_chats())
            return

        if path.startswith("/api/chats/"):
            chat_id = path[len("/api/chats/") :].strip("/")
            if not chat_id or "/" in chat_id:
                self._not_found("invalid chat id")
                return
            payload = load_chat(chat_id)
            if payload is None:
                self._not_found(f"chat '{chat_id}' not found")
                return
            self._json(200, payload)
            return

        # Static files
        if path in ("/", "/index.html"):
            rel = "index.html"
        else:
            rel = path.lstrip("/")

        target = (HERE / rel).resolve()
        if HERE not in target.parents and target != HERE:
            self._not_found()
            return
        if not target.is_file():
            self._not_found()
            return

        data = target.read_bytes()
        ctype, _ = mimetypes.guess_type(str(target))
        if ctype is None:
            ctype = "application/octet-stream"
        if ctype.startswith("text/") or ctype in (
            "application/javascript",
            "application/json",
        ):
            ctype = f"{ctype}; charset=utf-8"
        self._send(200, data, ctype)


def main() -> int:
    global SESSIONS_DIR, CHATS_DIR

    parser = argparse.ArgumentParser(description="Session / chat graph viewer")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help=f"Path to sessions folder (default: {DEFAULT_SESSIONS})",
    )
    parser.add_argument(
        "--chats-dir",
        type=Path,
        default=None,
        help=f"Path to chats folder (default: {DEFAULT_CHATS})",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    args = parser.parse_args()

    SESSIONS_DIR = (args.sessions_dir or DEFAULT_SESSIONS).resolve()
    CHATS_DIR = (args.chats_dir or DEFAULT_CHATS).resolve()
    if not SESSIONS_DIR.is_dir():
        print(
            f"error: sessions dir does not exist: {SESSIONS_DIR}\n"
            f"pass --sessions-dir PATH",
            file=sys.stderr,
        )
        return 1
    if not CHATS_DIR.is_dir():
        # Chats are optional for one-shot-only workflows; create empty hint.
        print(f"warning: chats dir missing ({CHATS_DIR}); chat dropdown will be empty",
              file=sys.stderr)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Ollive graph viewer")
    print(f"  sessions: {SESSIONS_DIR}")
    print(f"  chats:    {CHATS_DIR}")
    print(f"  open:     http://{args.host}:{args.port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
