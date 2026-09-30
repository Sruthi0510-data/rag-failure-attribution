"""Ranking chunks against a question, with a candidate pool wider than k on purpose.

The wider pool is what separates a rank miss from an embedding miss. Without it, every failure
below k looks the same, and "add a reranker" and "change the model" become indistinguishable
proposals, which is how a retrieval roadmap ends up being a matter of taste.

Ties are broken by chunk id rather than left to the sort's input order. Two chunks with equal
similarity are common in a lexical index, and a rank that depends on dictionary insertion order
would make every published rank number a fact about one interpreter run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rfa.chunking import Chunk
from rfa.embed import TfidfEmbedder, cosine
from rfa.errors import ConfigError

POOL_MULTIPLE = 4
"""How much wider than k the candidate pool runs. Four is the usual reranker fan-out."""


@dataclass(frozen=True)
class Ranking:
    """The ranked chunk ids for one question, and the rank lookup that goes with it."""

    question_id: str
    ranked_ids: tuple[str, ...]
    scores: tuple[float, ...]

    def rank_of(self, chunk_id: str) -> int | None:
        """One-based rank, or None when the chunk is not in the ranking at all."""
        try:
            return self.ranked_ids.index(chunk_id) + 1
        except ValueError:
            return None


class Index:
    """A fitted embedder plus the encoded chunks. Built once and queried many times."""

    def __init__(self, chunks: Sequence[Chunk], embedder: TfidfEmbedder | None = None) -> None:
        if not chunks:
            raise ConfigError("cannot build an index over zero chunks")
        self.chunks = tuple(chunks)
        self.embedder = embedder or TfidfEmbedder()
        if not self.embedder.fitted:
            self.embedder.fit([chunk.text for chunk in self.chunks])
        self.vectors = {chunk.chunk_id: self.embedder.encode(chunk.text) for chunk in self.chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def rank(self, question_id: str, query: str, limit: int | None = None) -> Ranking:
        """Rank every chunk, or the top `limit`, most similar first."""
        vector = self.embedder.encode(query)
        scored = [
            (cosine(vector, self.vectors[chunk.chunk_id]), chunk.chunk_id) for chunk in self.chunks
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        if limit is not None:
            scored = scored[:limit]
        return Ranking(
            question_id=question_id,
            ranked_ids=tuple(chunk_id for _, chunk_id in scored),
            scores=tuple(score for score, _ in scored),
        )


def pool_size(k: int, multiple: int = POOL_MULTIPLE) -> int:
    """How many candidates a reranker would realistically see, given a target k."""
    if k < 1:
        raise ConfigError("k must be at least 1")
    if multiple < 1:
        raise ConfigError("the pool multiple must be at least 1")
    return k * multiple
