"""The corpus model, and the two span predicates the whole repository turns on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfa.corpus import (
    AnswerSpan,
    Corpus,
    Document,
    load_corpus,
    total_characters,
    write_corpus,
)
from rfa.errors import CorpusError
from tests.conftest import doc, question


def test_a_span_reports_its_length():
    assert AnswerSpan("d", 10, 25).length == 15


def test_a_negative_span_start_is_refused():
    with pytest.raises(CorpusError, match="negative"):
        AnswerSpan("d", -1, 5)


@pytest.mark.parametrize("end", [0, 5, 4])
def test_a_span_that_does_not_advance_is_refused(end):
    with pytest.raises(CorpusError, match="not after start"):
        AnswerSpan("d", 5, end)


def test_a_chunk_containing_a_span_also_overlaps_it():
    span = AnswerSpan("d", 10, 20)
    assert span.contained_by(0, 30) is True
    assert span.overlaps(0, 30) is True


def test_a_chunk_holding_half_an_answer_overlaps_and_does_not_contain():
    """The distinction the whole repository rests on.

    A chunk that overlaps a span holds part of the answer. Part of an answer does not answer
    the question, and recall@k cannot tell the two apart because both are the right chunk id.
    """
    span = AnswerSpan("d", 10, 20)
    assert span.overlaps(0, 15) is True
    assert span.contained_by(0, 15) is False


def test_a_chunk_ending_exactly_where_a_span_starts_does_not_overlap_it():
    assert AnswerSpan("d", 10, 20).overlaps(0, 10) is False


def test_a_chunk_starting_exactly_where_a_span_ends_does_not_overlap_it():
    assert AnswerSpan("d", 10, 20).overlaps(20, 30) is False


def test_a_chunk_matching_a_span_exactly_contains_it():
    assert AnswerSpan("d", 10, 20).contained_by(10, 20) is True


def test_a_chunk_one_character_short_does_not_contain_the_span():
    assert AnswerSpan("d", 10, 20).contained_by(10, 19) is False


def test_a_document_reports_its_length_and_slices():
    document = doc("d", "abcdefgh")
    assert document.length == 8
    assert document.slice(2, 5) == "cde"


def test_a_corpus_refuses_duplicate_document_ids():
    with pytest.raises(CorpusError, match="duplicate doc_id"):
        Corpus(documents=(doc("d"), doc("d")), questions=())


def test_a_corpus_refuses_duplicate_question_ids():
    with pytest.raises(CorpusError, match="duplicate question_id"):
        Corpus(documents=(doc("d1"),), questions=(question("q", doc_id="d1"),) * 2)


def test_a_question_annotating_an_unknown_document_is_refused():
    with pytest.raises(CorpusError, match="unknown document"):
        Corpus(documents=(doc("d1"),), questions=(question("q1", doc_id="d9"),))


def test_a_span_running_past_the_end_of_its_document_is_refused():
    """Caught at construction, because the alternative is a silent absent classification later."""
    with pytest.raises(CorpusError, match="characters long"):
        Corpus(documents=(doc("d1", "abc"),), questions=(question("q1", doc_id="d1", end=99),))


def test_a_span_ending_exactly_at_the_document_end_is_accepted():
    Corpus(documents=(doc("d1", "abc"),), questions=(question("q1", doc_id="d1", start=0, end=3),))


def test_answer_text_returns_the_annotated_characters(tiny_corpus):
    assert tiny_corpus.answer_text(tiny_corpus.questions[0]) == "alpha beta"


def test_the_mean_answer_length_is_rounded_to_one_place(tiny_corpus):
    assert tiny_corpus.mean_answer_length == 10.0


def test_an_empty_question_set_has_a_mean_answer_length_of_zero():
    assert Corpus(documents=(doc(),), questions=()).mean_answer_length == 0.0


def test_by_id_indexes_every_document(tiny_corpus):
    assert set(tiny_corpus.by_id) == {"d1", "d2"}


def test_total_characters_sums_document_lengths(tiny_corpus):
    assert total_characters(tiny_corpus.documents) == sum(d.length for d in tiny_corpus.documents)


def test_a_corpus_round_trips_through_json(tmp_path: Path, tiny_corpus):
    path = tmp_path / "c.json"
    assert write_corpus(path, tiny_corpus) == 2
    reloaded = load_corpus(path)
    assert reloaded.documents == tiny_corpus.documents
    assert reloaded.questions == tiny_corpus.questions


def test_a_missing_corpus_names_the_path(tmp_path: Path):
    with pytest.raises(CorpusError, match="corpus not found"):
        load_corpus(tmp_path / "absent.json")


def test_invalid_json_is_reported_as_such(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusError, match="not valid JSON"):
        load_corpus(path)


@pytest.mark.parametrize("missing", ["documents", "questions"])
def test_a_corpus_missing_a_top_level_key_is_refused(tmp_path: Path, missing):
    payload = {"documents": [], "questions": []}
    payload.pop(missing)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorpusError, match=missing):
        load_corpus(path)


def test_writing_a_corpus_creates_missing_parent_directories(tmp_path: Path, tiny_corpus):
    path = tmp_path / "a" / "b" / "c.json"
    write_corpus(path, tiny_corpus)
    assert path.exists()


def test_a_document_title_survives_a_round_trip(tmp_path: Path):
    corpus = Corpus(documents=(Document("d1", "abc", title="Policy"),), questions=())
    path = tmp_path / "c.json"
    write_corpus(path, corpus)
    assert load_corpus(path).documents[0].title == "Policy"
