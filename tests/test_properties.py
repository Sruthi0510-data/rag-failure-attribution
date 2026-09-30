"""The claims the README makes, asserted over the shipped corpus.

The numbers themselves are guarded by tools/check_numbers.py. These tests guard the shape of
the claims: that the two recall definitions diverge, that the divergence is not a budget
problem, and that the closed form describes uniform placement and only approximates the corpus.
"""

from __future__ import annotations

import random

import pytest

from rfa.chunking import (
    CHUNKER_GEOMETRY,
    fixed_size,
    get_chunker,
    predicted_severance_rate,
)
from rfa.corpus import AnswerSpan, Document
from rfa.evaluate import evaluate
from rfa.failures import FailureClass

DEFAULT = "fixed-320"


@pytest.fixture(scope="module")
def baseline(shipped_corpus):
    return evaluate(shipped_corpus, get_chunker(DEFAULT), DEFAULT, k=5)


def test_the_usual_metric_reports_a_healthy_number_on_an_unhealthy_pipeline(baseline):
    """The headline, as one assertion. Both halves are true at once."""
    assert baseline.rate(baseline.recall_generous) > 80.0
    assert baseline.rate(baseline.answerable) < 50.0


def test_most_scored_hits_cannot_actually_be_answered(baseline):
    assert baseline.false_hits > baseline.recall_strict / 2


def test_most_recorded_misses_are_not_a_retrieval_problem(baseline):
    assert baseline.severed_misses > baseline.retrieval_fixable


def test_chunking_work_outweighs_retrieval_work_on_this_corpus(baseline):
    assert baseline.chunking_fixable > baseline.retrieval_fixable


def test_raising_k_cannot_raise_the_answerable_ceiling(shipped_corpus):
    """The impossibility, measured. The ceiling is a property of the boundaries, not of k."""
    ceilings = {
        evaluate(shipped_corpus, get_chunker(DEFAULT), DEFAULT, k=k, pool=max(k * 4, k)).answerable
        for k in (1, 5, 20, 100, 200)
    }
    assert len(ceilings) == 1


def test_overlap_based_recall_reaches_a_perfect_score_while_the_ceiling_stands_still(
    shipped_corpus,
):
    result = evaluate(shipped_corpus, get_chunker(DEFAULT), DEFAULT, k=200, pool=800)
    assert result.rate(result.recall_generous) == 100.0
    assert result.rate(result.answerable) < 100.0


def test_every_point_of_recall_gained_past_the_ceiling_is_a_false_hit(shipped_corpus):
    low = evaluate(shipped_corpus, get_chunker(DEFAULT), DEFAULT, k=20, pool=80)
    high = evaluate(shipped_corpus, get_chunker(DEFAULT), DEFAULT, k=200, pool=800)
    assert high.recall_strict == low.recall_strict
    assert high.false_hits > low.false_hits


def test_a_smaller_chunk_size_severs_more_answers(shipped_corpus):
    small = evaluate(shipped_corpus, get_chunker("fixed-160"), "fixed-160", k=5)
    large = evaluate(shipped_corpus, get_chunker("fixed-640"), "fixed-640", k=5)
    assert small.answerable < large.answerable


def test_overlap_recovers_answers_and_adds_rank_misses(shipped_corpus):
    """Overlap does not remove a failure, it converts one into a cheaper failure."""
    plain = evaluate(shipped_corpus, get_chunker("fixed-320"), "fixed-320", k=5)
    overlapping = evaluate(
        shipped_corpus, get_chunker("fixed-320-overlap-80"), "fixed-320-overlap-80", k=5
    )

    def ranked(result):
        return sum(
            1
            for outcome in result.outcomes
            if outcome.failure in {FailureClass.RANK_MISS, FailureClass.EMBEDDING_MISS}
        )

    assert overlapping.answerable > plain.answerable
    assert ranked(overlapping) > ranked(plain)
    assert overlapping.index_multiple > plain.index_multiple


def test_a_sentence_aligned_chunker_still_severs_some_answers(shipped_corpus):
    """Because a share of answers span two sentences, which is what a real annotation does."""
    result = evaluate(shipped_corpus, get_chunker("sentence-3"), "sentence-3", k=5)
    assert result.total - result.answerable > 0


def test_the_closed_form_describes_uniform_placement_within_two_points():
    rng = random.Random(20260930)
    document = Document(doc_id="synthetic", text="x" * 20000)
    worst = 0.0
    for size, stride in sorted(set(CHUNKER_GEOMETRY.values())):
        chunks = fixed_size(size, stride)([document])
        for length in (150, 210, 300):
            severed = 0
            for _ in range(600):
                start = rng.randrange(0, document.length - length)
                span = AnswerSpan("synthetic", start, start + length)
                if not any(span.contained_by(c.start, c.end) for c in chunks):
                    severed += 1
            measured = 100.0 * severed / 600
            predicted = 100.0 * predicted_severance_rate(length, size, stride)
            worst = max(worst, abs(measured - predicted))
    assert worst < 6.0


def test_the_closed_form_only_approximates_the_shipped_corpus(shipped_corpus):
    """Span placement in the corpus is not uniform, and the residual says by how much."""
    lengths = [question.answer.length for question in shipped_corpus.questions]
    residuals = []
    for name, (size, stride) in CHUNKER_GEOMETRY.items():
        result = evaluate(shipped_corpus, get_chunker(name), name, k=5)
        measured = 100.0 * (1.0 - result.answerable / result.total)
        predicted = (
            100.0
            * sum(predicted_severance_rate(length, size, stride) for length in lengths)
            / len(lengths)
        )
        residuals.append(abs(measured - predicted))
    assert max(residuals) > 4.0


def test_no_chunker_makes_every_question_answerable_and_every_answer_retrievable(shipped_corpus):
    """There is no free configuration on this corpus, which is why the trade-off table exists."""
    perfect = []
    for name in ("fixed-320", "fixed-640", "sentence-3", "sentence-3-overlap-1", "paragraph"):
        result = evaluate(shipped_corpus, get_chunker(name), name, k=5)
        if result.answerable == result.total and result.recall_strict == result.total:
            perfect.append(name)
    assert perfect == []
