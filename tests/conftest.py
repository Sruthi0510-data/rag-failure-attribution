"""Shared builders, so the corpus schema lives in one place."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfa.corpus import AnswerSpan, Corpus, Document, Question, load_corpus
from rfa.generator import GeneratorConfig, generate

CORPUS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "policy_corpus.json"


def doc(doc_id: str = "d1", text: str = "abcdefghij", title: str = "") -> Document:
    return Document(doc_id=doc_id, text=text, title=title)


def question(
    question_id: str = "q1",
    text: str = "what",
    doc_id: str = "d1",
    start: int = 0,
    end: int = 3,
) -> Question:
    return Question(
        question_id=question_id, text=text, answer=AnswerSpan(doc_id=doc_id, start=start, end=end)
    )


@pytest.fixture
def tiny_corpus() -> Corpus:
    """Two short documents and two questions, small enough to reason about by hand."""
    return Corpus(
        documents=(
            doc("d1", "alpha beta gamma delta epsilon zeta eta theta"),
            doc("d2", "iota kappa lambda mu nu xi omicron pi rho sigma"),
        ),
        questions=(
            question("q1", "alpha beta", "d1", 0, 10),
            question("q2", "omicron pi", "d2", 22, 32),
        ),
    )


@pytest.fixture
def small_corpus() -> Corpus:
    """A deterministic 12-document corpus, fast enough for per-test evaluation."""
    return generate(GeneratorConfig(documents=12, questions=24, seed=7))


@pytest.fixture(scope="session")
def shipped_corpus() -> Corpus:
    """The committed corpus the README quotes. Session scoped: read once."""
    return load_corpus(CORPUS_PATH)
