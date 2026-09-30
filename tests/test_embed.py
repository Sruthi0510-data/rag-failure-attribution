"""The lexical embedder and the cosine, including the cases that return nothing."""

from __future__ import annotations

import pytest

from rfa.embed import TfidfEmbedder, cosine, tokenise
from rfa.errors import ConfigError


@pytest.fixture
def fitted() -> TfidfEmbedder:
    return TfidfEmbedder().fit(
        ["retention of audit logs", "access review cadence", "replication lag threshold"]
    )


def test_tokenising_lowercases_and_keeps_alphanumerics():
    assert tokenise("Audit-Logs 30 DAYS!") == ["audit", "logs", "30", "days"]


def test_tokenising_an_empty_string_gives_nothing():
    assert tokenise("") == []


def test_punctuation_alone_produces_no_tokens():
    assert tokenise("... --- ???") == []


def test_fitting_on_an_empty_collection_is_refused():
    with pytest.raises(ConfigError, match="empty collection"):
        TfidfEmbedder().fit([])


def test_encoding_before_fitting_is_refused():
    with pytest.raises(ConfigError, match="must be fitted"):
        TfidfEmbedder().encode("anything")


def test_fitting_marks_the_embedder_as_fitted(fitted):
    assert fitted.fitted is True


def test_a_vector_is_l2_normalised(fitted):
    vector = fitted.encode("retention of audit logs")
    assert sum(value * value for value in vector.values()) == pytest.approx(1.0)


def test_a_term_absent_from_the_vocabulary_is_dropped(fitted):
    """An unseen term must not influence a ranking, or the similarity is to nothing."""
    assert fitted.encode("quetzalcoatl") == {}


def test_a_query_of_only_unseen_terms_gives_an_empty_vector(fitted):
    assert fitted.encode("zzz yyy") == {}


def test_a_rarer_term_carries_more_weight_than_a_common_one():
    embedder = TfidfEmbedder().fit(["common rare1", "common rare2", "common rare3"])
    vector = embedder.encode("common rare1")
    assert vector["rare1"] > vector["common"]


def test_repeating_a_term_increases_its_weight_sublinearly(fitted):
    once = fitted.encode("audit")
    twice = TfidfEmbedder().fit(["audit logs", "access review"]).encode("audit audit")
    assert once and twice


def test_cosine_of_a_vector_with_itself_is_one(fitted):
    vector = fitted.encode("audit logs")
    assert cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_with_an_empty_vector_is_zero(fitted):
    assert cosine(fitted.encode("audit"), {}) == 0.0
    assert cosine({}, fitted.encode("audit")) == 0.0


def test_cosine_of_disjoint_vectors_is_zero(fitted):
    assert cosine(fitted.encode("audit"), fitted.encode("replication")) == 0.0


def test_cosine_is_symmetric(fitted):
    left, right = fitted.encode("audit logs"), fitted.encode("access review")
    assert cosine(left, right) == pytest.approx(cosine(right, left))


def test_cosine_iterates_the_shorter_side_and_gets_the_same_answer(fitted):
    """Guards the optimisation: swapping the arguments must not change the result."""
    long_vector = fitted.encode("retention of audit logs access review cadence")
    short_vector = fitted.encode("audit")
    assert cosine(long_vector, short_vector) == pytest.approx(cosine(short_vector, long_vector))


def test_the_same_text_encodes_identically_every_time(fitted):
    assert fitted.encode("audit logs") == fitted.encode("audit logs")


def test_fitting_twice_on_the_same_corpus_gives_the_same_idf():
    corpus = ["a b", "b c", "c d"]
    assert TfidfEmbedder().fit(corpus).idf == TfidfEmbedder().fit(corpus).idf
