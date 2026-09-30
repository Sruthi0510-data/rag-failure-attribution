"""Chunkers, and the closed form checked against brute force at small sizes."""

from __future__ import annotations

import pytest

from rfa.chunking import (
    CHUNKER_GEOMETRY,
    CHUNKERS,
    Chunk,
    fixed_size,
    get_chunker,
    index_size,
    overlap_needed,
    paragraph,
    predicted_severance_rate,
    sentence_window,
    split_sentences,
)
from rfa.corpus import AnswerSpan
from rfa.errors import ConfigError
from tests.conftest import doc


def test_fixed_chunks_cover_the_whole_document():
    chunks = fixed_size(4)([doc("d", "abcdefghij")])
    assert "".join(chunk.text for chunk in chunks) == "abcdefghij"


def test_fixed_chunks_are_contiguous_without_overlap():
    chunks = fixed_size(4)([doc("d", "abcdefghij")])
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (4, 8), (8, 10)]


def test_the_last_fixed_chunk_is_truncated_at_the_document_end():
    chunks = fixed_size(4)([doc("d", "abcdefghij")])
    assert chunks[-1].length == 2


def test_overlapping_chunks_repeat_characters():
    chunks = fixed_size(4, stride=2)([doc("d", "abcdefgh")])
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (2, 6), (4, 8)]


def test_chunk_ids_are_unique_within_a_document():
    chunks = fixed_size(3)([doc("d", "a" * 30)])
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_ids_carry_the_document_they_came_from():
    chunks = fixed_size(3)([doc("alpha", "a" * 9)])
    assert all(chunk.chunk_id.startswith("alpha#") for chunk in chunks)


def test_every_chunk_text_matches_its_offsets():
    document = doc("d", "the quick brown fox jumps over the lazy dog")
    for chunk in fixed_size(7, stride=5)([document]):
        assert chunk.text == document.slice(chunk.start, chunk.end)


@pytest.mark.parametrize("size", [0, -1])
def test_a_non_positive_chunk_size_is_refused(size):
    with pytest.raises(ConfigError, match="at least 1"):
        fixed_size(size)


def test_a_non_positive_stride_is_refused():
    with pytest.raises(ConfigError, match="at least 1"):
        fixed_size(10, stride=0)


def test_a_stride_above_the_size_is_refused_because_it_would_skip_characters():
    with pytest.raises(ConfigError, match="no chunk at all"):
        fixed_size(10, stride=11)


def test_sentence_splitting_covers_every_character():
    text = "One. Two! Three? Four"
    spans = split_sentences(text)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_sentence_splitting_keeps_the_whitespace_between_sentences():
    """A splitter that drops the gap creates characters in no chunk, which reads as absent."""
    text = "One.  Two."
    assert "".join(text[a:b] for a, b in split_sentences(text)) == text


def test_a_document_with_no_terminator_is_one_sentence():
    assert split_sentences("no terminator here") == [(0, 18)]


def test_an_empty_document_still_yields_one_span():
    assert split_sentences("") == [(0, 0)]


def test_a_sentence_window_groups_the_requested_number_of_sentences():
    chunks = sentence_window(2)([doc("d", "A. B. C. D.")])
    assert len(chunks) == 2


def test_a_sentence_window_with_overlap_repeats_a_sentence():
    chunks = sentence_window(2, overlap=1)([doc("d", "A. B. C.")])
    assert len(chunks) == 2
    assert chunks[0].end > chunks[1].start


@pytest.mark.parametrize("sentences", [0, -1])
def test_a_sentence_window_needs_at_least_one_sentence(sentences):
    with pytest.raises(ConfigError, match="at least one sentence"):
        sentence_window(sentences)


def test_a_negative_sentence_overlap_is_refused():
    with pytest.raises(ConfigError, match="cannot be negative"):
        sentence_window(3, overlap=-1)


def test_an_overlap_equal_to_the_window_is_refused_because_it_never_advances():
    with pytest.raises(ConfigError, match="never advances"):
        sentence_window(3, overlap=3)


def test_paragraph_chunks_split_on_blank_lines():
    chunks = paragraph()([doc("d", "first block\n\nsecond block")])
    assert [chunk.text for chunk in chunks] == ["first block", "second block"]


def test_paragraph_offsets_point_at_the_original_text():
    document = doc("d", "first block\n\nsecond block")
    for chunk in paragraph()([document]):
        assert document.slice(chunk.start, chunk.end) == chunk.text


def test_an_empty_paragraph_is_skipped_rather_than_indexed():
    chunks = paragraph()([doc("d", "first\n\n\n\nsecond")])
    assert len(chunks) == 2


def test_index_size_sums_chunk_lengths():
    chunks = [Chunk("a", "d", 0, 10, "x" * 10), Chunk("b", "d", 10, 15, "x" * 5)]
    assert index_size(chunks) == 15


def test_overlap_inflates_the_index_beyond_the_corpus():
    document = doc("d", "x" * 1000)
    plain = index_size(fixed_size(100)([document]))
    overlapping = index_size(fixed_size(100, stride=50)([document]))
    assert overlapping > plain


def test_the_closed_form_matches_brute_force_at_a_small_size():
    """Checked exhaustively rather than by sampling, at a size where that is affordable."""
    size, stride, length = 12, 12, 5
    document = doc("d", "x" * 600)
    chunks = fixed_size(size, stride)([document])
    severed = 0
    positions = range(0, document.length - length)
    for start in positions:
        span = AnswerSpan("d", start, start + length)
        if not any(span.contained_by(c.start, c.end) for c in chunks):
            severed += 1
    measured = severed / len(positions)
    assert abs(measured - predicted_severance_rate(length, size, stride)) < 0.02


def test_severance_is_zero_when_an_answer_is_one_character():
    assert predicted_severance_rate(1, 100, 100) == 0.0


def test_severance_doubles_when_the_chunk_size_halves():
    assert predicted_severance_rate(51, 200, 200) == pytest.approx(
        predicted_severance_rate(51, 400, 400) * 2, rel=0.01
    )


def test_an_answer_longer_than_a_chunk_is_always_severed():
    assert predicted_severance_rate(500, 320, 320) == 1.0


def test_an_answer_exactly_as_long_as_a_chunk_is_almost_always_severed():
    assert predicted_severance_rate(320, 320, 320) > 0.99


def test_overlap_of_length_minus_one_removes_severance_entirely():
    length = 60
    size = 320
    stride = size - overlap_needed(length)
    assert predicted_severance_rate(length, size, stride) == 0.0


def test_one_character_less_overlap_does_not_quite_remove_severance():
    length = 60
    size = 320
    stride = size - (overlap_needed(length) - 1)
    assert predicted_severance_rate(length, size, stride) > 0.0


def test_overlap_needed_is_one_short_of_the_answer_length():
    assert overlap_needed(60) == 59


@pytest.mark.parametrize("bad", [0, -3])
def test_the_closed_form_refuses_a_non_positive_span_length(bad):
    with pytest.raises(ConfigError, match="at least 1"):
        predicted_severance_rate(bad, 100, 100)


@pytest.mark.parametrize("args", [(50, 0, 10), (50, 10, 0)])
def test_the_closed_form_refuses_non_positive_geometry(args):
    with pytest.raises(ConfigError, match="at least 1"):
        predicted_severance_rate(*args)


def test_the_closed_form_refuses_a_stride_above_the_size():
    with pytest.raises(ConfigError, match="leave gaps"):
        predicted_severance_rate(50, 100, 200)


def test_overlap_needed_refuses_a_non_positive_length():
    with pytest.raises(ConfigError, match="at least 1"):
        overlap_needed(0)


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_every_registered_chunker_produces_chunks(name, tiny_corpus):
    assert get_chunker(name)(tiny_corpus.documents)


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_every_registered_chunker_keeps_chunk_text_consistent_with_offsets(name, small_corpus):
    by_id = small_corpus.by_id
    for chunk in get_chunker(name)(small_corpus.documents):
        assert by_id[chunk.doc_id].slice(chunk.start, chunk.end) == chunk.text


def test_an_unknown_chunker_lists_the_valid_names():
    with pytest.raises(ConfigError, match="paragraph"):
        get_chunker("no-such-chunker")


def test_only_arithmetic_chunkers_carry_a_geometry():
    """Sentence and paragraph boundaries are a property of the text, not of a progression.

    Listing them in CHUNKER_GEOMETRY would let the closed form be applied where its uniform
    placement assumption has no meaning, which is the sort of tidy number this repository
    exists to argue against.
    """
    assert set(CHUNKER_GEOMETRY) <= set(CHUNKERS)
    assert not any(name.startswith(("sentence", "paragraph")) for name in CHUNKER_GEOMETRY)
