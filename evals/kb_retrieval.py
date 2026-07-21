"""KB chunking + fuzzy keyword retrieval for the hallucination judge.

Mirrors the agent's sliding-window chunker (mcp_server._chunk_text) and
stopword tokenization (memory._tokens), then scores chunks by keyword
overlap with a difflib fuzzy match for near-misses.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

# Default: sibling agent/sandbox/kb at repo root, resolved from this file.
_HERE = Path(__file__).resolve().parent
DEFAULT_KB_DIR = (_HERE.parent / "agent" / "sandbox" / "kb").resolve()

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
WORD_BUDGET = 12_000
FUZZY_CUTOFF = 0.85

_STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "or", "in", "on", "for", "at",
    "with", "by", "from", "what", "how", "when", "where", "why", "this", "that",
    "it", "be", "as", "are", "was", "were", "i", "you", "me", "my", "your",
}


@dataclass
class KbChunk:
    source: str
    text: str
    index: int
    tokens: set[str] = field(default_factory=set)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunking by word count — same contract as the agent."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    stride = max(1, size - overlap)
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += stride
    return chunks


def tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"\w+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def load_kb(kb_dir: Path | None = None) -> list[KbChunk]:
    """Load and chunk every *.md under kb_dir."""
    root = Path(kb_dir) if kb_dir else DEFAULT_KB_DIR
    out: list[KbChunk] = []
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        for i, chunk in enumerate(chunk_text(text)):
            out.append(KbChunk(
                source=rel,
                text=chunk,
                index=i,
                tokens=tokens(chunk),
            ))
    return out


def _token_hits(query_toks: set[str], chunk_toks: set[str]) -> int:
    """Exact overlap + fuzzy close-matches against the chunk vocabulary."""
    if not query_toks or not chunk_toks:
        return 0
    score = 0
    chunk_list = list(chunk_toks)
    for q in query_toks:
        if q in chunk_toks:
            score += 1
            continue
        if difflib.get_close_matches(q, chunk_list, n=1, cutoff=FUZZY_CUTOFF):
            score += 1
    return score


def retrieve_chunks(
    query: str,
    *,
    kb: list[KbChunk] | None = None,
    kb_dir: Path | None = None,
    word_budget: int = WORD_BUDGET,
) -> list[dict]:
    """Return scored KB chunks relevant to `query`, capped by word_budget.

    Each item: {source, index, text, score, word_count}.
    """
    corpus = kb if kb is not None else load_kb(kb_dir)
    qtoks = tokens(query)
    if not qtoks or not corpus:
        return []

    scored: list[tuple[int, KbChunk]] = []
    for c in corpus:
        s = _token_hits(qtoks, c.tokens)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: (-x[0], x[1].source, x[1].index))

    out: list[dict] = []
    used = 0
    for s, c in scored:
        wc = len(c.text.split())
        if used + wc > word_budget and out:
            break
        if used + wc > word_budget and not out:
            # Always include at least one chunk even if it alone exceeds budget.
            pass
        out.append({
            "source": c.source,
            "index": c.index,
            "text": c.text,
            "score": s,
            "word_count": wc,
        })
        used += wc
        if used >= word_budget:
            break
    return out


def format_kb_section(chunks: list[dict]) -> str:
    if not chunks:
        return "(no matching knowledge-base chunks)"
    parts: list[str] = []
    for c in chunks:
        parts.append(
            f"[{c['source']} chunk {c['index'] + 1}  score={c['score']}]\n{c['text']}"
        )
    return "\n\n".join(parts)
