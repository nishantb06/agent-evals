"""On-disk persistence for the growing graph.

Lives in its own file because flow.py needs to stay under 350 lines.
The two surfaces:

  - SessionStore: per-session directory under state/sessions/<sid>/.
    Owns reading and writing the graph pickle and the per-node JSON
    files. Atomic-write semantics (write to tmp, rename) so a SIGKILL
    mid-write does not corrupt the last successful snapshot.
  - rebuild_graph_state(): given a populated SessionStore, returns the
    list of NodeState records sorted by completion time so replay.py
    can walk them in order.

The Graph itself (the NetworkX wrapping) lives in flow.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx

from schemas import AgentResult, NodeState

SESSIONS_ROOT = Path(__file__).parent / "state" / "sessions"


class SessionLoadError(RuntimeError):
    """Raised when a persisted session cannot be safely loaded.

    Examples: a NodeState file that no longer matches the schema, a
    `_result_typed` payload that cannot round-trip back into an
    AgentResult. We fail loud here rather than silently degrade — the
    Executor's downstream code does `isinstance(..., AgentResult)`
    checks, and stashing a dict where it expects a Pydantic model is
    silent data corruption — fail loud rather than degrade silently."""


def _atomic_write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


class SessionStore:
    """One on-disk session. Layout:

        state/sessions/<sid>/
            graph.pkl              # NetworkX DiGraph pickle
            query.txt              # the user's verbatim query
            nodes/
                n_001.json         # NodeState for the n:1 node, etc.
                n_002.json
                ...
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = SESSIONS_ROOT / session_id
        self.nodes_dir = self.dir / "nodes"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)

    @property
    def query_path(self) -> Path:
        return self.dir / "query.txt"

    @property
    def graph_path(self) -> Path:
        # Graph is persisted as JSON via nx.node_link_data so the file
        # is human-readable and the format survives a Python upgrade.
        return self.dir / "graph.json"

    @property
    def _legacy_graph_path(self) -> Path:
        # Older sessions wrote pickle; the loader tolerates that for resume
        # on pre-fix sessions but the writer always emits JSON now.
        return self.dir / "graph.pkl"

    def write_query(self, query: str) -> None:
        _atomic_write(self.query_path, query)

    def read_query(self) -> str:
        if not self.query_path.exists():
            return ""
        return self.query_path.read_text()

    def write_graph(self, graph_obj: nx.DiGraph) -> None:
        """Serialise the DiGraph to JSON via nx.node_link_data. Per-node
        `result` is an AgentResult (Pydantic) — dump it to a dict so the
        JSON encoder is happy. Reviving on read restores the Pydantic shape.
        """
        # node_link_data accepts arbitrary node-attr dicts; we just need
        # every value to be JSON-serialisable.
        h = nx.DiGraph()
        for n, d in graph_obj.nodes(data=True):
            attrs = dict(d)
            if isinstance(attrs.get("result"), AgentResult):
                attrs["result"] = attrs["result"].model_dump(mode="json")
                attrs["_result_typed"] = True
            h.add_node(n, **attrs)
        for u, v, d in graph_obj.edges(data=True):
            h.add_edge(u, v, **d)
        payload = nx.node_link_data(h, edges="edges")
        _atomic_write(self.graph_path, json.dumps(payload, indent=2, default=str))

    def read_graph(self) -> nx.DiGraph | None:
        if self.graph_path.exists():
            payload = json.loads(self.graph_path.read_text())
            g = nx.node_link_graph(payload, edges="edges", directed=True)
            # A write tagged a node's `result` as a typed AgentResult via
            # `_result_typed`. If the dict no longer round-trips through
            # AgentResult.model_validate, that is silent data corruption —
            # raise instead. SessionLoadError surfaces the bad file path
            # and the validation message so the operator can act on it.
            for nid, d in g.nodes(data=True):
                if d.pop("_result_typed", False) and isinstance(d.get("result"), dict):
                    try:
                        d["result"] = AgentResult.model_validate(d["result"])
                    except (ValueError, TypeError) as e:
                        raise SessionLoadError(
                            f"node {nid} in {self.graph_path}: persisted "
                            f"AgentResult failed model_validate. The graph "
                            f"is unsafe to resume — inspect the file and "
                            f"either repair it or delete the session. "
                            f"validation error: {type(e).__name__}: {e}"
                        ) from e
            return g
        # Backwards-compat: tolerate sessions written by the older pickle
        # path. We import pickle lazily so the dependency is only paid when
        # someone resumes a legacy session.
        if self._legacy_graph_path.exists():
            import pickle, sys
            print(f"[persistence] reading legacy pickle graph from "
                  f"{self._legacy_graph_path}", file=sys.stderr)
            return pickle.loads(self._legacy_graph_path.read_bytes())
        return None

    def _node_path(self, node_id: str) -> Path:
        # node_id is like "n:1" — turn that into n_001.json so directory
        # listings sort sensibly.
        try:
            i = int(node_id.split(":", 1)[1])
            return self.nodes_dir / f"n_{i:03d}.json"
        except (IndexError, ValueError):
            safe = node_id.replace(":", "_").replace("/", "_")
            return self.nodes_dir / f"{safe}.json"

    def write_node(self, state: NodeState) -> None:
        _atomic_write(self._node_path(state.node_id), state.model_dump_json(indent=2))

    def write_memory_hits(self, hits: list) -> None:
        """Persist FAISS/memory hits for this graph run (viewer + debugging).

        Each hit is a plain dict: id, kind, descriptor, source, chunk, raw.
        Similarity scores are not available from memory.read today.
        """
        payload: list[dict] = []
        for h in hits or []:
            if hasattr(h, "model_dump"):
                d = h.model_dump(mode="json")
            elif isinstance(h, dict):
                d = h
            else:
                d = {
                    "id": getattr(h, "id", None),
                    "kind": getattr(h, "kind", None),
                    "descriptor": getattr(h, "descriptor", None),
                    "source": getattr(h, "source", None),
                    "value": getattr(h, "value", {}) or {},
                }
            val = d.get("value") or {}
            payload.append({
                "id": d.get("id"),
                "kind": d.get("kind"),
                "descriptor": d.get("descriptor") or "",
                "source": d.get("source") or "",
                "chunk": val.get("chunk") or "",
                "raw": val.get("raw") or "",
            })
        _atomic_write(
            self.dir / "memory_hits.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

    def read_node(self, node_id: str) -> NodeState | None:
        p = self._node_path(node_id)
        if not p.exists():
            return None
        return NodeState.model_validate_json(p.read_text())

    def read_all_nodes(self) -> list[NodeState]:
        """Load every persisted NodeState in this session. Corrupt or
        partially-written files (the typical cause is a process kill between
        the temp-file write and the atomic rename) are skipped with a clear
        warning to stderr — never silently dropped."""
        import sys
        states: list[NodeState] = []
        for p in sorted(self.nodes_dir.glob("n_*.json")):
            try:
                states.append(NodeState.model_validate_json(p.read_text()))
            except (OSError, ValueError) as e:
                # OSError = unreadable; ValueError covers JSON decode +
                # Pydantic ValidationError (which inherits ValueError).
                print(f"[persistence] WARNING: skipped corrupt node file "
                      f"{p}: {type(e).__name__}: {e}", file=sys.stderr)
        return states


def list_sessions() -> list[str]:
    if not SESSIONS_ROOT.exists():
        return []
    return sorted(p.name for p in SESSIONS_ROOT.iterdir() if p.is_dir())
