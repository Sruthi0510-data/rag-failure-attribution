"""Chunkers, and the arithmetic that predicts how many answers each one will cut in half.

The theory function is the reason this module is worth reading. Severance is not a mysterious
property of a corpus: for a fixed-size chunker it follows from three numbers, and the closed
form is derivable in four lines.

A chunk k covers `[k*stride, k*stride + size)`. A span `[p, p + length)` is contained by chunk
k exactly when `k*stride <= p` and `p + length <= k*stride + size`, so the chunk starts that
work form the interval `[p + length - size, p]`, which is `size - length` wide. For p uniform
over the document, that interval contains a valid chunk start with probability
`min(1, (size - length + 1) / stride)`.

Two consequences fall straight out. With no overlap the severance rate is `(length - 1) / size`,
so halving the chunk size doubles it. And severance reaches zero exactly when the overlap
reaches `length - 1`, which is a fact about the longest answer in the evaluation set rather
than about the retriever. The experiments measure the real rate against this prediction, and
the fact that they agree is what turns the finding from an observation into a model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from rfa.corpus import Document
from rfa.errors import ConfigError

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A half-open character range of one document, with its text carried along."""

    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start


Chunker = Callable[[Sequence[Document]], list[Chunk]]


def predicted_severance_rate(span_length: int, size: int, stride: int) -> float:
    """Expected share of uniformly placed spans that no single chunk contains.

    Returns 1.0 when a span is longer than a chunk, because then no chunk can contain it at
    any offset, which is the degenerate case a chunk-size sweep walks into at the small end.
    """
    if span_length < 1:
        raise ConfigError("span_length must be at least 1")
    if size < 1 or stride < 1:
        raise ConfigError("size and stride must be at least 1")
    if stride > size:
        raise ConfigError("stride above size would leave gaps between chunks")
    if span_length > size:
        return 1.0
    return round(max(0.0, 1.0 - min(1.0, (size - span_length + 1) / stride)), 4)


def overlap_needed(span_length: int) -> int:
    """The overlap at which severance reaches zero: one character short of the longest span.

    A fact about the evaluation set, not about the retriever, which is why no amount of model
    tuning moves it.
    """
    if span_length < 1:
        raise ConfigError("span_length must be at least 1")
    return span_length - 1


def fixed_size(size: int, stride: int | None = None) -> Chunker:
    """Characters, the chunker every tutorial starts with. `stride` defaults to no overlap."""
    stride = size if stride is None else stride
    if size < 1:
        raise ConfigError("chunk size must be at least 1")
    if stride < 1:
        raise ConfigError("stride must be at least 1")
    if stride > size:
        raise ConfigError("stride above size would leave characters in no chunk at all")

    def chunk(documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            start = 0
            index = 0
            while start < document.length:
                end = min(start + size, document.length)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#c{index}",
                        doc_id=document.doc_id,
                        start=start,
                        end=end,
                        text=document.slice(start, end),
                    )
                )
                index += 1
                if end >= document.length:
                    break
                start += stride
        return chunks

    return chunk


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries as character ranges, covering the whole document with no gaps.

    Deliberately simple and deliberately total. A smarter splitter would handle abbreviations
    better; what matters here is that every character belongs to exactly one sentence, because
    a splitter that drops whitespace between sentences creates uncovered characters and a span
    landing there would be reported as absent rather than severed, which is a different and
    much more alarming finding.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in SENTENCE_END.finditer(text):
        spans.append((cursor, match.end()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans or [(0, len(text))]


def sentence_window(sentences: int, overlap: int = 0) -> Chunker:
    """Whole sentences, which is what a team switches to after the first severance complaint.

    It reduces severance for answers that are one sentence long and does nothing at all for an
    answer that spans two, which the experiments measure rather than assert.
    """
    if sentences < 1:
        raise ConfigError("a sentence window needs at least one sentence")
    if overlap < 0:
        raise ConfigError("overlap cannot be negative")
    if overlap >= sentences:
        raise ConfigError("overlap must be smaller than the window, or the window never advances")

    def chunk(documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            boundaries = split_sentences(document.text)
            step = sentences - overlap
            index = 0
            position = 0
            while position < len(boundaries):
                window = boundaries[position : position + sentences]
                start, end = window[0][0], window[-1][1]
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#s{index}",
                        doc_id=document.doc_id,
                        start=start,
                        end=end,
                        text=document.slice(start, end),
                    )
                )
                index += 1
                if position + sentences >= len(boundaries):
                    break
                position += step
        return chunks

    return chunk


def paragraph() -> Chunker:
    """Blank-line separated blocks. The least severance and the worst retrieval precision."""

    def chunk(documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            cursor = 0
            index = 0
            for block in document.text.split("\n\n"):
                start = cursor
                end = start + len(block)
                cursor = end + 2
                if not block.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#p{index}",
                        doc_id=document.doc_id,
                        start=start,
                        end=end,
                        text=block,
                    )
                )
                index += 1
        return chunks

    return chunk


CHUNKERS: dict[str, Chunker] = {
    "fixed-320": fixed_size(320),
    "fixed-320-overlap-80": fixed_size(320, stride=240),
    "fixed-160": fixed_size(160),
    "fixed-640": fixed_size(640),
    "sentence-3": sentence_window(3),
    "sentence-3-overlap-1": sentence_window(3, overlap=1),
    "paragraph": paragraph(),
}

CHUNKER_GEOMETRY: dict[str, tuple[int, int]] = {
    "fixed-320": (320, 320),
    "fixed-320-overlap-80": (320, 240),
    "fixed-160": (160, 160),
    "fixed-640": (640, 640),
}
"""Size and stride for the chunkers the closed form applies to.

Sentence and paragraph chunkers are absent on purpose. Their boundaries are a property of the
text rather than of an arithmetic progression, so the uniform-placement model does not describe
them and pretending otherwise would be the sort of tidy number this repository exists to argue
against.
"""


def get_chunker(name: str) -> Chunker:
    """Look up a chunker, failing with the list of valid names rather than a KeyError."""
    try:
        return CHUNKERS[name]
    except KeyError as exc:
        raise ConfigError(f"unknown chunker {name!r}; available: {sorted(CHUNKERS)}") from exc


def index_size(chunks: Iterable[Chunk]) -> int:
    """Total characters stored across all chunks, which overlap inflates and nobody budgets."""
    return sum(chunk.length for chunk in chunks)
