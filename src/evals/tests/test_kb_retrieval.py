"""Tests for kb_retrieval chunking, fuzzy matching, and budget cap."""

from __future__ import annotations

from kb_retrieval import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    KbChunk,
    chunk_text,
    retrieve_chunks,
    tokens,
)


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_parity_with_agent_logic() -> None:
    words = [f"w{i}" for i in range(950)]
    text = " ".join(words)
    chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    # Same stride formula the agent uses: size - overlap
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    expected = 1
    i = 0
    while i + CHUNK_SIZE < len(words):
        i += stride
        expected += 1
    assert len(chunks) == expected
    assert chunks[0].split()[0] == "w0"
    assert len(chunks[0].split()) == CHUNK_SIZE


def test_tokens_filters_stopwords() -> None:
    toks = tokens("The supplements and the diet for you")
    assert "the" not in toks
    assert "and" not in toks
    assert "for" not in toks
    assert "you" not in toks
    assert "supplements" in toks
    assert "diet" in toks


def test_fuzzy_matching_hits_typo() -> None:
    corpus = [
        KbChunk(
            source="08_Natural_Supplements.md",
            text="Ashwagandha and turmeric are common natural supplements for stress.",
            index=0,
            tokens=tokens("Ashwagandha and turmeric are common natural supplements for stress."),
        ),
        KbChunk(
            source="01_Diet.md",
            text="Whole foods and vegetables form the basis of a healthy diet pattern.",
            index=0,
            tokens=tokens("Whole foods and vegetables form the basis of a healthy diet pattern."),
        ),
    ]
    # typo: supplments ≈ supplements
    hits = retrieve_chunks("any particular natural supplments that can help", kb=corpus)
    assert hits
    assert hits[0]["source"] == "08_Natural_Supplements.md"
    assert hits[0]["score"] > 0


def test_budget_cap_limits_words() -> None:
    corpus = []
    for i in range(20):
        text = " ".join([f"token{i}"] + ["padding"] * 100)
        corpus.append(KbChunk(
            source=f"doc{i}.md",
            text=text,
            index=0,
            tokens=tokens(text),
        ))
    hits = retrieve_chunks("token5 token7 token9", kb=corpus, word_budget=250)
    assert hits
    total = sum(h["word_count"] for h in hits)
    # May slightly exceed only if first chunk alone is huge; our chunks are ~101 words.
    assert total <= 250 + 101
