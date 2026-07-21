"""Parse MEMORY HITS blocks from skill prompt_sent text.

The agent embeds FAISS-ranked hits in prompts via skills._format_memory_hits:

  MEMORY HITS (N from FAISS):
    - [fact] <descriptor>
        source: ...
        chunk: ...
"""

from __future__ import annotations

import re
from typing import Any

_HEADER_RE = re.compile(r"^MEMORY HITS \(\d+ from FAISS\):\s*$", re.MULTILINE)
# Top-level sections that follow the hits block in render_prompt.
_STOP_RE = re.compile(
    r"^(CHAT HISTORY|USER_QUERY:|QUESTION:|FAILURE:|PERSONA:|INPUTS:|MEMORY HITS )",
    re.MULTILINE,
)
_HIT_START_RE = re.compile(r"^  - \[([^\]]+)\] (.*)$", re.MULTILINE)


def parse_memory_hits_from_prompt(prompt: str | None) -> list[dict[str, Any]]:
    """Extract structured hits from a prompt_sent string. Returns [] if absent."""
    if not prompt or "MEMORY HITS (" not in prompt:
        return []

    # Prefer the last MEMORY HITS block — earlier mentions appear in planner.md
    # instructions; the real payload is injected near the end of the prompt.
    matches = list(_HEADER_RE.finditer(prompt))
    if not matches:
        return []
    start = matches[-1].end()
    rest = prompt[start:]
    stop = _STOP_RE.search(rest)
    block = rest[: stop.start()] if stop else rest

    hits: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in block.splitlines():
        m = _HIT_START_RE.match(line)
        if m:
            if current:
                hits.append(current)
            current = {
                "kind": m.group(1).strip(),
                "descriptor": m.group(2).strip(),
                "source": "",
                "chunk": "",
                "raw": "",
            }
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("source:"):
            current["source"] = stripped[len("source:") :].strip()
        elif stripped.startswith("chunk:"):
            current["chunk"] = stripped[len("chunk:") :].strip()
        elif stripped.startswith("raw:"):
            current["raw"] = stripped[len("raw:") :].strip()
        elif current.get("chunk") and line.startswith("      ") and not stripped.startswith(
            ("source:", "chunk:", "raw:")
        ):
            # Continuation lines for long chunks (rare; usually single-line).
            current["chunk"] += " " + stripped
        elif current.get("raw") and line.startswith("      "):
            current["raw"] += " " + stripped

    if current:
        hits.append(current)
    return hits


def extract_memory_hits_from_nodes(nodes: dict[str, dict]) -> list[dict[str, Any]]:
    """Prefer planner prompt_sent; fall back to any node that has a hits block."""
    ordered = sorted(
        nodes.values(),
        key=lambda n: (
            0 if (n.get("skill") == "planner") else 1,
            n.get("node_id") or n.get("id") or "",
        ),
    )
    for n in ordered:
        hits = parse_memory_hits_from_prompt(n.get("prompt_sent"))
        if hits:
            return hits
    return []


def extract_retriever_chunks(nodes: dict[str, dict]) -> list[dict[str, Any]]:
    """Collect chunks from retriever skill outputs."""
    out: list[dict[str, Any]] = []
    for n in nodes.values():
        if n.get("skill") != "retriever":
            continue
        result = n.get("result") or {}
        output = result.get("output") or {}
        chunks = output.get("chunks")
        if isinstance(chunks, list):
            for c in chunks:
                if isinstance(c, dict):
                    out.append(c)
                elif isinstance(c, str):
                    out.append({"preview": c})
        summary = output.get("summary")
        if summary and not out:
            out.append({"summary": summary})
    return out
