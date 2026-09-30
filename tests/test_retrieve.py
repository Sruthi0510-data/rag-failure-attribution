"""Ranking, tie breaking and the candidate pool that separates two retrieval failures."""

from __future__ import annotations

import pytest

from rfa.chunking import Chunk, fixed_size
from rfa.errors import ConfigError
from rfa.retrieve import POOL_MULTIPLE, Index, Ranking, pool_size
from tests.conftest import doc


@pytest.fixture
def index() -> Index:
    document = doc(
        "d",
        "retention of audit logs is thirty days. "
        "access reviews run every ninety days. "
        "replication lag pages the on-call engineer. ",
    )
    return Index(fixed_size(40)([document]))


def test_an_index_over_zero_chunks_is_refused():
    with pytest.raises(ConfigError, match="zero chunks"):
        Index([])


def test_an_index_reports_how_many_chunks_it_holds(index):
    assert len(index) == len(index.chunks)


def test_ranking_returns_every_chunk_when_no_limit_is_given(index):
    assert len(index.rank("q", "audit").ranked_ids) == len(index)


def test_a_limit_truncates_the_ranking(index):
    assert len(index.rank("q", "audit", limit=2).ranked_ids) == 2


def test_the_ranking_carries_the_question_id(index):
    assert index.rank("q7", "audit").question_id == "q7"


def test_scores_are_non_increasing_down_the_ranking(index):
    scores = index.rank("q", "audit logs retention").scores
    assert list(scores) == sorted(scores, reverse=True)


def test_a_matching_query_ranks_the_matching_chunk_first(index):
    top = index.rank("q", "pages engineer on call").ranked_ids[0]
    by_id = {chunk.chunk_id: chunk.text for chunk in index.chunks}
    assert "engineer" in by_id[top]


def test_a_character_chunker_cuts_words_as_readily_as_it_cuts_answers(index):
    """Found by accident while writing the test above, and kept because it is the thesis.

    The 40-character boundary lands inside the word "replication", so the chunk that should be
    the obvious match for a replication query does not contain the term at all. The same
    mechanism that severs an answer severs the token that would have retrieved it, which is why
    a low recall number and a chunking problem look identical from the outside.
    """
    texts = [chunk.text for chunk in index.chunks]
    assert any("plication" in text and "replication" not in text for text in texts)


def test_a_query_with_no_known_terms_still_returns_a_full_ranking(index):
    ranking = index.rank("q", "quetzalcoatl")
    assert len(ranking.ranked_ids) == len(index)
    assert set(ranking.scores) == {0.0}


def test_ties_are_broken_by_chunk_id_rather_than_by_insertion_order():
    """A rank that depended on dictionary order would make every published rank a fluke."""
    chunks = [
        Chunk("d#z", "d", 0, 5, "alpha"),
        Chunk("d#a", "d", 5, 10, "alpha"),
        Chunk("d#m", "d", 10, 15, "alpha"),
    ]
    ranking = Index(chunks).rank("q", "alpha")
    assert ranking.ranked_ids == ("d#a", "d#m", "d#z")


def test_ranking_is_reproducible(index):
    assert index.rank("q", "audit").ranked_ids == index.rank("q", "audit").ranked_ids


def test_rank_of_returns_a_one_based_position(index):
    ranking = index.rank("q", "audit")
    assert ranking.rank_of(ranking.ranked_ids[0]) == 1
    assert ranking.rank_of(ranking.ranked_ids[2]) == 3


def test_rank_of_returns_none_for_a_chunk_outside_the_ranking():
    ranking = Ranking(question_id="q", ranked_ids=("a", "b"), scores=(1.0, 0.5))
    assert ranking.rank_of("c") is None


def test_the_pool_runs_wider_than_k_by_the_documented_multiple():
    assert pool_size(5) == 5 * POOL_MULTIPLE


def test_the_pool_multiple_can_be_overridden():
    assert pool_size(5, multiple=2) == 10


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_k_is_refused(bad):
    with pytest.raises(ConfigError, match="k must be at least 1"):
        pool_size(bad)


def test_a_non_positive_pool_multiple_is_refused():
    with pytest.raises(ConfigError, match="multiple must be at least 1"):
        pool_size(5, multiple=0)


def test_the_inverted_index_ranks_identically_to_the_exhaustive_scan(index):
    """The optimisation is only worth having if it changes nothing, so this pins that.

    Tie ordering is included deliberately: the fast path appends the zero-scoring chunks in
    chunk id order, which is what the exhaustive sort produced for them anyway. If that ever
    diverges, every published rank number becomes a fact about which code path ran.
    """
    for query in ("audit logs", "replication lag", "quetzalcoatl", "thirty days on-call"):
        fast = index.rank("q", query)
        slow = index.rank_exhaustively("q", query)
        assert fast.ranked_ids == slow.ranked_ids
        assert fast.scores == pytest.approx(slow.scores)


def test_the_two_paths_agree_under_a_limit(index):
    for limit in (1, 2, len(index)):
        assert index.rank("q", "audit", limit=limit).ranked_ids == index.rank_exhaustively(
            "q", "audit", limit=limit
        ).ranked_ids


def test_the_inverted_index_holds_a_posting_list_per_term(index):
    assert index.postings
    assert all(postings for postings in index.postings.values())


def test_a_term_absent_from_the_corpus_has_no_posting_list(index):
    assert "quetzalcoatl" not in index.postings


def test_an_index_accepts_a_prefitted_embedder(index):
    from rfa.embed import TfidfEmbedder

    embedder = TfidfEmbedder().fit(["audit logs", "replication lag"])
    reused = Index(index.chunks, embedder=embedder)
    assert reused.embedder is embedder
