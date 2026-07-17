"""Structure-aware recursive chunking.

Split on the largest natural boundary that fits (paragraph, then sentence, then
word), so chunks rarely cut mid-thought. Overlap carries context across the seam
so a fact split across a boundary is still retrievable from either side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    start: int
    end: int

    def as_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    if not seps:
        # No boundary left: hard-cut on size.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = seps[0], seps[1:]
    pieces = text.split(sep)
    out: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = piece if not buf else buf + sep + piece
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(piece) > size:
            out.extend(_split_recursive(piece, size, rest))
            buf = ""
        else:
            buf = piece
    if buf.strip():
        out.append(buf)
    return [p for p in out if p.strip()]


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Return non-empty chunks of roughly `size` chars with `overlap` carry-over."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []

    pieces = _split_recursive(text, size, _SEPARATORS)

    chunks: list[Chunk] = []
    cursor = 0
    carry = ""
    for piece in pieces:
        body = (carry + " " + piece).strip() if carry else piece
        start = text.find(piece, cursor)
        if start == -1:
            start = cursor
        end = start + len(piece)
        chunks.append(Chunk(len(chunks), body, start, end))
        cursor = end
        carry = piece[-overlap:] if overlap and len(piece) > overlap else piece if overlap else ""
    return chunks
