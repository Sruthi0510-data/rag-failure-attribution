"""The evaluation engine: classify every question, then report what recall@k got wrong.

Three things are computed per question and they are deliberately kept apart.

Answerability asks whether any chunk in the whole index contains the annotated span. It has
nothing to do with retrieval and it is the ceiling every retrieval metric is measured against.

Generous recall@k asks whether a chunk overlapping the span made the top k. This is what an
evaluation harness computes in practice, because the ground truth it was handed is a document
id or a passage and overlap is the only comparison available.

Strict recall@k asks whether a chunk containing the whole span made the top k. The gap between
the two is the set of questions the usual metric scores as wins and the pipeline cannot answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rfa.chunking import Chunk, Chunker, index_size
from rfa.corpus import Corpus
from rfa.errors import ConfigError
from rfa.failures import (
    CHUNKING_FIXABLE,
    RETRIEVAL_FIXABLE,
    FailureClass,
    QuestionOutcome,
)
from rfa.retrieve import Index, pool_size


def classify(
    corpus: Corpus,
    chunks: Sequence[Chunk],
    index: Index,
    k: int,
    pool: int,
) -> list[QuestionOutcome]:
    """Work out, for each question, which of the six classes it falls into."""
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    outcomes: list[QuestionOutcome] = []
    for question in corpus.questions:
        span = question.answer
        candidates = by_doc.get(span.doc_id, [])
        containing = [c for c in candidates if span.contained_by(c.start, c.end)]
        overlapping = [c for c in candidates if span.overlaps(c.start, c.end)]

        ranking = index.rank(question.question_id, question.text, limit=pool)
        containing_ranks = [r for r in (ranking.rank_of(c.chunk_id) for c in containing) if r]
        overlapping_ranks = [r for r in (ranking.rank_of(c.chunk_id) for c in overlapping) if r]
        best_containing = min(containing_ranks) if containing_ranks else None
        best_overlapping = min(overlapping_ranks) if overlapping_ranks else None

        failure = _classify_one(
            containing=bool(containing),
            overlapping=bool(overlapping),
            best_containing=best_containing,
            best_overlapping=best_overlapping,
            k=k,
        )
        outcomes.append(
            QuestionOutcome(
                question_id=question.question_id,
                failure=failure,
                answerable=bool(containing),
                best_containing_rank=best_containing,
                best_overlapping_rank=best_overlapping,
                containing_chunks=len(containing),
                overlapping_chunks=len(overlapping),
                answer_length=span.length,
                spans_chunks=len(overlapping),
            )
        )
    return outcomes


def _classify_one(
    containing: bool,
    overlapping: bool,
    best_containing: int | None,
    best_overlapping: int | None,
    k: int,
) -> FailureClass:
    """The decision, kept in one place so the taxonomy is auditable at a glance.

    Order matters. Absence is checked before severance because a span no chunk touches is an
    ingestion problem rather than a boundary problem, and reporting it as severed would send
    somebody to tune a chunk size that was never involved.
    """
    if not overlapping:
        return FailureClass.ABSENT
    if not containing:
        scored_hit = best_overlapping is not None and best_overlapping <= k
        return FailureClass.FALSE_HIT if scored_hit else FailureClass.SEVERED
    if best_containing is not None and best_containing <= k:
        return FailureClass.HIT
    if best_containing is not None:
        return FailureClass.RANK_MISS
    return FailureClass.EMBEDDING_MISS


@dataclass(frozen=True)
class Evaluation:
    """One chunker, one k, and every rate the documents quote, derived rather than stored."""

    chunker: str
    k: int
    pool: int
    outcomes: tuple[QuestionOutcome, ...]
    chunk_count: int
    index_characters: int
    corpus_characters: int
    mean_answer_length: float

    @property
    def total(self) -> int:
        return len(self.outcomes)

    def _count(self, *classes: FailureClass) -> int:
        wanted = set(classes)
        return sum(1 for outcome in self.outcomes if outcome.failure in wanted)

    def rate(self, count: int, base: int | None = None) -> float:
        base = self.total if base is None else base
        return 0.0 if base == 0 else round(100.0 * count / base, 1)

    @property
    def answerable(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.answerable)

    @property
    def recall_generous(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.scored_hit_generous)

    @property
    def recall_strict(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.scored_hit_strict)

    @property
    def false_hits(self) -> int:
        """Scored as wins by the usual metric, with the answer unrecoverable from the index."""
        return self._count(FailureClass.FALSE_HIT)

    @property
    def misses(self) -> int:
        """Questions the usual metric records as retrieval failures."""
        return self.total - self.recall_generous

    @property
    def severed_misses(self) -> int:
        return self._count(FailureClass.SEVERED)

    @property
    def retrieval_fixable(self) -> int:
        return self._count(*RETRIEVAL_FIXABLE)

    @property
    def chunking_fixable(self) -> int:
        return self._count(*CHUNKING_FIXABLE)

    @property
    def index_multiple(self) -> float:
        """How many times the corpus the index stores. Above one means overlap is being paid for."""
        if self.corpus_characters == 0:
            return 0.0
        return round(self.index_characters / self.corpus_characters, 2)

    def breakdown(self) -> dict[str, int]:
        return {
            str(cls): self._count(cls)
            for cls in FailureClass
            if self._count(cls) or cls is FailureClass.HIT
        }

    def summary(self) -> dict[str, Any]:
        return {
            "chunker": self.chunker,
            "k": self.k,
            "pool": self.pool,
            "questions": self.total,
            "chunks": self.chunk_count,
            "index_multiple": self.index_multiple,
            "mean_answer_length": self.mean_answer_length,
            "answerable": self.answerable,
            "answerable_pct": self.rate(self.answerable),
            "recall_generous": self.recall_generous,
            "recall_generous_pct": self.rate(self.recall_generous),
            "recall_strict": self.recall_strict,
            "recall_strict_pct": self.rate(self.recall_strict),
            "false_hits": self.false_hits,
            "false_hit_share_of_scored_hits_pct": self.rate(self.false_hits, self.recall_generous),
            "misses": self.misses,
            "severed_misses": self.severed_misses,
            "severed_share_of_misses_pct": self.rate(self.severed_misses, self.misses),
            "retrieval_fixable": self.retrieval_fixable,
            "chunking_fixable": self.chunking_fixable,
            "breakdown": self.breakdown(),
        }


def evaluate(
    corpus: Corpus,
    chunker: Chunker,
    chunker_name: str,
    k: int = 5,
    pool: int | None = None,
) -> Evaluation:
    """Chunk, index, rank and classify. The one entry point the CLI and experiments share."""
    if k < 1:
        raise ConfigError("k must be at least 1")
    chunks = chunker(corpus.documents)
    if not chunks:
        raise ConfigError(f"chunker {chunker_name!r} produced no chunks")
    pool = pool_size(k) if pool is None else pool
    if pool < k:
        raise ConfigError(f"a candidate pool of {pool} cannot contain the top {k}")
    index = Index(chunks)
    outcomes = classify(corpus, chunks, index, k=k, pool=pool)
    return Evaluation(
        chunker=chunker_name,
        k=k,
        pool=pool,
        outcomes=tuple(outcomes),
        chunk_count=len(chunks),
        index_characters=index_size(chunks),
        corpus_characters=sum(document.length for document in corpus.documents),
        mean_answer_length=corpus.mean_answer_length,
    )
