"""Chunking is where RAG quality is silently won or lost, so it gets real tests."""

import pytest

from app.rag.chunker import chunk_text


def test_short_text_is_one_chunk():
    chunks = chunk_text("Hello world.", size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].chunk_index == 0


def test_empty_text_yields_nothing():
    assert chunk_text("   \n\n  ") == []


def test_chunks_are_indexed_contiguously():
    text = " ".join(f"sentence number {i}." for i in range(300))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_no_empty_chunks_are_emitted():
    text = "para one.\n\n\n\n\npara two.\n\n\n" + ("x " * 500)
    assert all(c.text.strip() for c in chunk_text(text, size=100, overlap=20))


def test_overlap_carries_context_across_the_seam():
    text = "alpha beta gamma. " * 60
    chunks = chunk_text(text, size=120, overlap=40)
    assert len(chunks) > 1
    # Later chunks are padded with carry-over, so they exceed the raw split size.
    assert any(len(c.text) > 120 for c in chunks[1:])


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("some text here", size=100, overlap=100)


def test_prefers_paragraph_boundaries_over_mid_word_cuts():
    text = "First paragraph body.\n\nSecond paragraph body.\n\nThird paragraph body."
    chunks = chunk_text(text, size=30, overlap=0)
    assert all(not c.text.startswith(" ") for c in chunks)


def test_hard_cut_when_no_separator_exists():
    text = "a" * 500  # single unbreakable token
    chunks = chunk_text(text, size=100, overlap=0)
    assert len(chunks) >= 5
