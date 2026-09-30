"""The failure taxonomy, which is the thing recall@k structurally cannot produce.

recall@k answers one question: was a chunk the evaluation considers relevant in the top k. It
is computed over chunk ids. The ground truth is a character range in a document. The map
between them is many-to-many and lossy in the direction the metric reads it, so a miss carries
no information about which of four different things went wrong, and the four have different
owners.

SEVERED belongs to whoever chose the chunk size. RANK_MISS belongs to whoever owns the
reranker. EMBEDDING_MISS belongs to whoever owns the model or the query rewriting. ABSENT
belongs to whoever owns the ingestion. Handing all four to the retrieval team as "recall is
low" is how a quarter gets spent on the wrong knob.

FALSE_HIT is the class nobody asks for. The question is counted as a success by the usual
overlap-based scoring, a retrieved chunk does touch the annotated answer, and no chunk anywhere
in the index holds the whole of it. The metric is reporting a win on a question the pipeline
cannot answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FailureClass(StrEnum):
    """What actually went wrong, and therefore who can fix it."""

    HIT = "hit"
    """A chunk containing the whole answer was retrieved in the top k. Nothing to fix."""

    FALSE_HIT = "false_hit"
    """Scored as a success on overlap, and no chunk in the index contains the whole answer.

    The chunker severed the answer and the retrieval metric cannot see it, because a chunk
    holding half the answer is still the right chunk id.
    """

    SEVERED = "severed"
    """No chunk in the index contains the whole answer, and the overlap score was a miss too.

    Fixed by re-chunking, and by nothing else. A better model retrieves the same broken pieces.
    """

    RANK_MISS = "rank_miss"
    """A containing chunk exists and ranked below k but inside the wider candidate pool.

    The recoverable class: a reranker over more candidates, or a larger k, addresses it.
    """

    EMBEDDING_MISS = "embedding_miss"
    """A containing chunk exists and never appeared even in the wider candidate pool.

    Fixed by the model, the query rewriting, or a hybrid lexical pass. Not by a reranker.
    """

    ABSENT = "absent"
    """No chunk covers the annotated characters at all, so ingestion dropped them."""


RETRIEVAL_FIXABLE = frozenset({FailureClass.RANK_MISS, FailureClass.EMBEDDING_MISS})
"""Classes a retrieval change can address. The point of the taxonomy is that this is a subset."""

CHUNKING_FIXABLE = frozenset({FailureClass.SEVERED, FailureClass.FALSE_HIT})
"""Classes only a chunking change can address, whatever the leaderboard says about the model."""

SCORED_AS_HIT = frozenset({FailureClass.HIT, FailureClass.FALSE_HIT})
"""What the usual overlap-based recall@k counts as a success."""


@dataclass(frozen=True)
class QuestionOutcome:
    """Everything the evaluation established about one question."""

    question_id: str
    failure: FailureClass
    answerable: bool
    best_containing_rank: int | None
    best_overlapping_rank: int | None
    containing_chunks: int
    overlapping_chunks: int
    answer_length: int
    spans_chunks: int

    @property
    def scored_hit_generous(self) -> bool:
        """What an overlap-based recall@k records for this question."""
        return self.failure in SCORED_AS_HIT

    @property
    def scored_hit_strict(self) -> bool:
        """What a recall@k that demands the whole answer in one chunk records."""
        return self.failure is FailureClass.HIT

    @property
    def misattributed(self) -> bool:
        """True when the usual metric points at the wrong team for this question."""
        return self.failure in CHUNKING_FIXABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "failure": str(self.failure),
            "answerable": self.answerable,
            "best_containing_rank": self.best_containing_rank,
            "best_overlapping_rank": self.best_overlapping_rank,
            "containing_chunks": self.containing_chunks,
            "overlapping_chunks": self.overlapping_chunks,
            "answer_length": self.answer_length,
            "spans_chunks": self.spans_chunks,
            "scored_hit_generous": self.scored_hit_generous,
            "scored_hit_strict": self.scored_hit_strict,
            "misattributed": self.misattributed,
        }
