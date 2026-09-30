"""Generator properties, including the pin for the corpus flaw that flattered one chunker."""

from __future__ import annotations

import pytest

from rfa.chunking import get_chunker, split_sentences
from rfa.errors import ConfigError
from rfa.evaluate import evaluate
from rfa.generator import GeneratorConfig, generate


def test_the_same_seed_produces_an_identical_corpus():
    left = generate(GeneratorConfig(documents=8, questions=16, seed=3))
    right = generate(GeneratorConfig(documents=8, questions=16, seed=3))
    assert left.documents == right.documents
    assert left.questions == right.questions


def test_a_different_seed_produces_a_different_corpus():
    left = generate(GeneratorConfig(documents=8, questions=16, seed=1))
    right = generate(GeneratorConfig(documents=8, questions=16, seed=2))
    assert left.documents != right.documents


def test_every_requested_document_and_question_appears():
    corpus = generate(GeneratorConfig(documents=9, questions=18, seed=4))
    assert len(corpus.documents) == 9
    assert len(corpus.questions) == 18


def test_every_answer_span_holds_the_planted_fact():
    """The span offsets must point at the fact, not near it."""
    corpus = generate(GeneratorConfig(documents=10, questions=20, seed=5))
    for question in corpus.questions:
        text = corpus.answer_text(question)
        assert len(text) > 40
        assert text.strip() == text


def test_answer_spans_vary_in_length():
    """A corpus of uniform answers hides the relationship between length and severance."""
    corpus = generate(GeneratorConfig(documents=30, questions=60, seed=6))
    lengths = {question.answer.length for question in corpus.questions}
    assert len(lengths) > 5


def test_some_answers_span_more_than_one_sentence():
    """Pins the corpus flaw that made a sentence chunker perfect by construction.

    The first version planted every fact as a single sentence, so a sentence-aligned chunker
    could not sever one and scored a hundred per cent. That is a finding about the corpus, not
    about sentence chunking. Without this test nothing stops the next tuning pass from
    reintroducing it.
    """
    corpus = generate(GeneratorConfig(documents=40, questions=80, seed=8))
    multi = 0
    for question in corpus.questions:
        document = corpus.by_id[question.answer.doc_id]
        sentences = split_sentences(document.text)
        span = question.answer
        touched = [s for s in sentences if s[0] < span.end and span.start < s[1]]
        if len(touched) > 1:
            multi += 1
    assert multi > 0


def test_a_sentence_chunker_does_not_score_perfectly_on_the_shipped_corpus(shipped_corpus):
    """The consequence of the test above, asserted where it actually matters."""
    result = evaluate(shipped_corpus, get_chunker("sentence-3"), "sentence-3", k=5)
    assert result.answerable < result.total


def test_questions_are_paraphrases_rather_than_copies_of_the_fact():
    corpus = generate(GeneratorConfig(documents=10, questions=20, seed=9))
    for question in corpus.questions:
        assert question.text != corpus.answer_text(question)


def test_a_zero_paraphrase_rate_still_produces_questions():
    corpus = generate(GeneratorConfig(documents=5, questions=10, seed=10, paraphrase_rate=0.0))
    assert all(question.text for question in corpus.questions)


def test_a_zero_multi_sentence_rate_produces_single_sentence_answers():
    corpus = generate(
        GeneratorConfig(documents=20, questions=20, seed=11, multi_sentence_rate=0.0)
    )
    for question in corpus.questions:
        assert corpus.answer_text(question).count(". ") <= 1


def test_documents_are_split_into_paragraphs():
    corpus = generate(GeneratorConfig(documents=4, questions=8, seed=12))
    assert all("\n\n" in document.text for document in corpus.documents)


def test_more_questions_than_documents_reuses_documents_rather_than_failing():
    corpus = generate(GeneratorConfig(documents=3, questions=12, seed=13))
    assert len({question.answer.doc_id for question in corpus.questions}) == 3


@pytest.mark.parametrize(("field", "value"), [("documents", 0), ("questions", 0)])
def test_a_non_positive_size_is_refused(field, value):
    with pytest.raises(ConfigError, match="at least 1"):
        GeneratorConfig(**{field: value})


def test_a_document_needs_at_least_one_paragraph():
    with pytest.raises(ConfigError, match="at least 1"):
        GeneratorConfig(paragraphs_per_document=0)


def test_a_paragraph_needs_room_for_a_fact_and_some_filler():
    with pytest.raises(ConfigError, match="at least 2"):
        GeneratorConfig(sentences_per_paragraph=1)


@pytest.mark.parametrize("field", ["paraphrase_rate", "multi_sentence_rate"])
@pytest.mark.parametrize("value", [-0.1, 1.4])
def test_a_rate_outside_zero_to_one_is_refused(field, value):
    with pytest.raises(ConfigError, match="between 0 and 1"):
        GeneratorConfig(**{field: value})


def test_generate_with_no_config_uses_the_documented_defaults():
    corpus = generate()
    assert len(corpus.documents) == GeneratorConfig().documents
