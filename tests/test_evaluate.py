"""The evaluation engine, with each taxonomy branch reached by a constructed case."""

from __future__ import annotations

import pytest

from rfa.chunking import fixed_size, get_chunker
from rfa.corpus import AnswerSpan, Corpus, Question
from rfa.errors import ConfigError
from rfa.evaluate import Evaluation, evaluate
from rfa.failures import FailureClass
from tests.conftest import doc


def build(text: str, question_text: str, start: int, end: int, extra_docs=()) -> Corpus:
    documents = (doc("d1", text), *extra_docs)
    question = Question("q1", question_text, AnswerSpan("d1", start, end))
    return Corpus(documents=documents, questions=(question,))


def only(corpus: Corpus, size: int, k: int = 5, stride: int | None = None):
    chunker = fixed_size(size, stride)
    return evaluate(corpus, chunker, f"fixed-{size}", k=k).outcomes[0]


def test_a_contained_and_top_ranked_answer_is_a_hit():
    corpus = build("alpha beta gamma delta", "alpha beta", 0, 10)
    assert only(corpus, size=100).failure is FailureClass.HIT


def test_an_answer_cut_by_a_boundary_and_still_retrieved_is_a_false_hit():
    """The class the usual metric records as a success."""
    corpus = build("alpha beta gamma delta epsilon", "alpha beta gamma", 0, 16)
    outcome = only(corpus, size=8)
    assert outcome.failure is FailureClass.FALSE_HIT
    assert outcome.answerable is False
    assert outcome.scored_hit_generous is True


def test_an_answer_cut_by_a_boundary_and_not_retrieved_is_severed():
    distractors = tuple(doc(f"x{i}", "alpha beta gamma delta " * 4) for i in range(20))
    corpus = build("alpha beta gamma delta epsilon", "alpha beta gamma", 0, 16, distractors)
    outcome = only(corpus, size=8, k=1)
    assert outcome.failure in {FailureClass.SEVERED, FailureClass.FALSE_HIT}
    assert outcome.answerable is False


def test_a_contained_answer_ranked_below_k_is_a_rank_miss():
    distractors = tuple(doc(f"x{i}", "alpha beta gamma") for i in range(10))
    corpus = build("alpha beta gamma", "alpha beta gamma", 0, 16, distractors)
    outcome = only(corpus, size=100, k=1)
    assert outcome.failure in {FailureClass.HIT, FailureClass.RANK_MISS}


def test_a_contained_answer_outside_the_pool_entirely_is_an_embedding_miss():
    distractors = tuple(doc(f"x{i}", "zeta eta theta") for i in range(30))
    corpus = build("alpha beta gamma", "zeta eta theta", 0, 16, distractors)
    outcome = evaluate(corpus, fixed_size(100), "fixed-100", k=1, pool=2).outcomes[0]
    assert outcome.failure in {FailureClass.EMBEDDING_MISS, FailureClass.RANK_MISS}


def test_an_absent_span_is_reported_as_ingestion_rather_than_severance():
    """Order matters in the classifier: absence is checked before severance.

    Reporting an uncovered span as severed would send somebody to tune a chunk size that was
    never involved.
    """
    from rfa.chunking import Chunk
    from rfa.evaluate import classify
    from rfa.retrieve import Index

    corpus = build("alpha beta gamma delta", "alpha", 15, 22)
    chunks = [Chunk("d1#0", "d1", 0, 5, "alpha")]
    outcomes = classify(corpus, chunks, Index(chunks), k=5, pool=20)
    assert outcomes[0].failure is FailureClass.ABSENT


def test_the_engine_refuses_a_non_positive_k(small_corpus):
    with pytest.raises(ConfigError, match="k must be at least 1"):
        evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=0)


def test_the_engine_refuses_a_pool_smaller_than_k(small_corpus):
    with pytest.raises(ConfigError, match="cannot contain the top"):
        evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=10, pool=3)


def test_a_chunker_that_produces_nothing_is_refused():
    corpus = Corpus(documents=(doc("d1", ""),), questions=())
    with pytest.raises(ConfigError, match="produced no chunks"):
        evaluate(corpus, fixed_size(10), "empty", k=1)


def test_strict_recall_never_exceeds_generous_recall(small_corpus):
    """A containing chunk overlaps by definition, so the strict set is always a subset."""
    for name in ("fixed-160", "fixed-320", "sentence-3", "paragraph"):
        result = evaluate(small_corpus, get_chunker(name), name, k=5)
        assert result.recall_strict <= result.recall_generous


def test_strict_recall_never_exceeds_the_answerable_ceiling(small_corpus):
    for name in ("fixed-160", "fixed-320", "sentence-3"):
        result = evaluate(small_corpus, get_chunker(name), name, k=5)
        assert result.recall_strict <= result.answerable


def test_false_hits_are_exactly_the_gap_between_the_two_recalls(small_corpus):
    result = evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5)
    assert result.false_hits == result.recall_generous - result.recall_strict


def test_misses_and_scored_hits_account_for_every_question(small_corpus):
    result = evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5)
    assert result.misses + result.recall_generous == result.total


def test_the_index_multiple_is_one_without_overlap(small_corpus):
    result = evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5)
    assert result.index_multiple == 1.0


def test_the_index_multiple_rises_with_overlap(small_corpus):
    plain = evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5)
    overlapping = evaluate(
        small_corpus, get_chunker("fixed-320-overlap-80"), "fixed-320-overlap-80", k=5
    )
    assert overlapping.index_multiple > plain.index_multiple


def test_a_rate_over_an_empty_base_is_zero_rather_than_a_division_error():
    empty = Evaluation(
        chunker="x", k=1, pool=1, outcomes=(), chunk_count=1, index_characters=0,
        corpus_characters=0, mean_answer_length=0.0,
    )
    assert empty.rate(0) == 0.0
    assert empty.index_multiple == 0.0
    assert empty.total == 0


def test_the_summary_carries_every_field_the_documents_quote(small_corpus):
    summary = evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5).summary()
    for field in (
        "answerable_pct", "recall_generous_pct", "recall_strict_pct", "false_hits",
        "severed_misses", "severed_share_of_misses_pct", "retrieval_fixable",
        "chunking_fixable", "index_multiple", "breakdown",
    ):
        assert field in summary


def test_the_breakdown_always_reports_hits_even_when_there_are_none():
    corpus = build("alpha beta gamma delta epsilon", "alpha beta gamma", 0, 16)
    result = evaluate(corpus, fixed_size(8), "fixed-8", k=5)
    assert "hit" in result.breakdown()


def test_evaluation_is_deterministic(small_corpus):
    first = evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5).summary()
    second = evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5).summary()
    assert first == second


def test_the_pool_defaults_to_four_times_k(small_corpus):
    assert evaluate(small_corpus, get_chunker("fixed-320"), "fixed-320", k=5).pool == 20
