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
    """A fitted embedder, the encoded chunks, and a posting list per term.

    The obvious implementation scores the query against every chunk. It is correct and it is
    quadratic in the shape this repository actually runs: several hundred questions against
    several hundred chunks, repeated once per chunker and once per point on a k sweep. The
    inverted index touches only the chunks that share a term with the query, which for a
    lexical retriever is a small fraction of the index.

    The optimisation is only worth having if it changes nothing else, so a test ranks the same
    queries both ways and asserts the orderings are identical, including the tie ordering.
    Chunks sharing no term with the query score zero and are appended in chunk id order, which
    is what the exhaustive version produced for them too.
    """

    def __init__(self, chunks: Sequence[Chunk], embedder: TfidfEmbedder | None = None) -> None:
        if not chunks:
            raise ConfigError("cannot build an index over zero chunks")
        self.chunks = tuple(chunks)
        self.embedder = embedder or TfidfEmbedder()
        if not self.embedder.fitted:
            self.embedder.fit([chunk.text for chunk in self.chunks])
        self.vectors = {chunk.chunk_id: self.embedder.encode(chunk.text) for chunk in self.chunks}
        self.postings: dict[str, list[tuple[str, float]]] = {}
        for chunk in self.chunks:
            for term, weight in self.vectors[chunk.chunk_id].items():
                self.postings.setdefault(term, []).append((chunk.chunk_id, weight))
        self._chunk_ids = tuple(sorted(chunk.chunk_id for chunk in self.chunks))

    def __len__(self) -> int:
        return len(self.chunks)

    def rank(self, question_id: str, query: str, limit: int | None = None) -> Ranking:
        """Rank the chunks, most similar first, breaking ties by chunk id."""
        vector = self.embedder.encode(query)
        accumulator: dict[str, float] = {}
        for term, weight in vector.items():
            for chunk_id, chunk_weight in self.postings.get(term, ()):
                accumulator[chunk_id] = accumulator.get(chunk_id, 0.0) + weight * chunk_weight

        scored = sorted(
            ((score, chunk_id) for chunk_id, score in accumulator.items() if score > 0.0),
            key=lambda pair: (-pair[0], pair[1]),
        )
        if limit is None or len(scored) < limit:
            matched = {chunk_id for _, chunk_id in scored}
            scored.extend(
                (0.0, chunk_id) for chunk_id in self._chunk_ids if chunk_id not in matched
            )
        if limit is not None:
            scored = scored[:limit]
        return Ranking(
            question_id=question_id,
            ranked_ids=tuple(chunk_id for _, chunk_id in scored),
            scores=tuple(score for score, _ in scored),
        )

    def rank_exhaustively(self, question_id: str, query: str, limit: int | None = None) -> Ranking:
        """The unoptimised ranking, kept so a test can hold the fast path to it."""
        vector = self.embedder.encode(query)
        scored = sorted(
            (
                (cosine(vector, self.vectors[chunk.chunk_id]), chunk.chunk_id)
                for chunk in self.chunks
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )
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
