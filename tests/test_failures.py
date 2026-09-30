"""The taxonomy: the sets it defines, and the outcome properties the report reads."""

from __future__ import annotations

import pytest

from rfa.failures import (
    CHUNKING_FIXABLE,
    RETRIEVAL_FIXABLE,
    SCORED_AS_HIT,
    FailureClass,
    QuestionOutcome,
)


def outcome(failure: FailureClass, **overrides) -> QuestionOutcome:
    base = {
        "question_id": "q1",
        "failure": failure,
        "answerable": failure in {FailureClass.HIT, FailureClass.RANK_MISS,
                                  FailureClass.EMBEDDING_MISS},
        "best_containing_rank": None,
        "best_overlapping_rank": None,
        "containing_chunks": 0,
        "overlapping_chunks": 1,
        "answer_length": 100,
        "spans_chunks": 1,
    }
    base.update(overrides)
    return QuestionOutcome(**base)


def test_the_two_fixable_sets_do_not_overlap():
    """The point of the taxonomy is that a failure has one owner, not two."""
    assert not (CHUNKING_FIXABLE & RETRIEVAL_FIXABLE)


def test_a_hit_belongs_to_neither_fixable_set():
    assert FailureClass.HIT not in CHUNKING_FIXABLE | RETRIEVAL_FIXABLE


def test_absence_is_an_ingestion_problem_and_belongs_to_neither_set():
    assert FailureClass.ABSENT not in CHUNKING_FIXABLE | RETRIEVAL_FIXABLE


def test_a_false_hit_is_scored_as_a_hit_by_the_usual_metric():
    assert FailureClass.FALSE_HIT in SCORED_AS_HIT


def test_a_severed_question_is_not_scored_as_a_hit():
    assert FailureClass.SEVERED not in SCORED_AS_HIT


def test_a_false_hit_is_chunking_work_despite_scoring_as_a_success():
    assert FailureClass.FALSE_HIT in CHUNKING_FIXABLE


@pytest.mark.parametrize("cls", [FailureClass.RANK_MISS, FailureClass.EMBEDDING_MISS])
def test_rank_and_embedding_misses_are_retrieval_work(cls):
    assert cls in RETRIEVAL_FIXABLE


def test_a_hit_scores_under_both_definitions():
    result = outcome(FailureClass.HIT)
    assert result.scored_hit_generous is True
    assert result.scored_hit_strict is True


def test_a_false_hit_scores_generously_and_not_strictly():
    """The gap between the two definitions is the whole finding, in one assertion."""
    result = outcome(FailureClass.FALSE_HIT)
    assert result.scored_hit_generous is True
    assert result.scored_hit_strict is False


@pytest.mark.parametrize(
    "cls",
    [FailureClass.SEVERED, FailureClass.RANK_MISS, FailureClass.EMBEDDING_MISS,
     FailureClass.ABSENT],
)
def test_every_other_class_scores_under_neither_definition(cls):
    result = outcome(cls)
    assert result.scored_hit_generous is False
    assert result.scored_hit_strict is False


@pytest.mark.parametrize("cls", sorted(CHUNKING_FIXABLE))
def test_chunking_classes_are_reported_as_misattributed(cls):
    assert outcome(cls).misattributed is True


@pytest.mark.parametrize("cls", sorted(RETRIEVAL_FIXABLE))
def test_retrieval_classes_are_not_misattributed(cls):
    """The usual metric points at the right team for these, so they are not the finding."""
    assert outcome(cls).misattributed is False


def test_a_hit_is_not_misattributed():
    assert outcome(FailureClass.HIT).misattributed is False


def test_an_outcome_serialises_every_field_the_json_publishes():
    payload = outcome(FailureClass.SEVERED).as_dict()
    for field in (
        "question_id", "failure", "answerable", "best_containing_rank", "best_overlapping_rank",
        "containing_chunks", "overlapping_chunks", "answer_length", "spans_chunks",
        "scored_hit_generous", "scored_hit_strict", "misattributed",
    ):
        assert field in payload


def test_the_failure_class_serialises_as_its_string_value():
    assert outcome(FailureClass.SEVERED).as_dict()["failure"] == "severed"


def test_every_class_has_a_docstring_explaining_who_fixes_it():
    for cls in FailureClass:
        assert FailureClass.__doc__
        assert cls.value
