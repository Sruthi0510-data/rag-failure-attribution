"""Documents, questions, and the annotation that makes the whole argument possible.

The important detail is where the ground truth lives. An `AnswerSpan` is a character range in
a document, not a chunk id. That is how annotation actually works: a human reads a document and
highlights the sentence that answers the question, long before anybody has chosen a chunk size.

Every later problem follows from that. Chunking happens after annotation, it is a choice the
evaluation set knows nothing about, and the map from spans to chunks is many-to-many. A span
can land inside one chunk, straddle two, or be spread over three, and recall@k has no way to
express the difference because it only ever sees chunk ids.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rfa.errors import CorpusError


@dataclass(frozen=True)
class AnswerSpan:
    """A half-open character range `[start, end)` in one document."""

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise CorpusError(f"span start {self.start} is negative")
        if self.end <= self.start:
            raise CorpusError(f"span end {self.end} is not after start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, start: int, end: int) -> bool:
        """True when this span shares at least one character with `[start, end)`."""
        return self.start < end and start < self.end

    def contained_by(self, start: int, end: int) -> bool:
        """True when `[start, end)` holds every character of this span.

        The distinction from `overlaps` is the whole repository. A chunk that overlaps a span
        holds part of the answer, and part of an answer does not answer the question.
        """
        return start <= self.start and self.end <= end


@dataclass(frozen=True)
class Document:
    """One document, with its text and the sentence offsets a sentence chunker needs."""

    doc_id: str
    text: str
    title: str = ""

    @property
    def length(self) -> int:
        return len(self.text)

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]


@dataclass(frozen=True)
class Question:
    """A question and the span that answers it, annotated on the document."""

    question_id: str
    text: str
    answer: AnswerSpan

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "doc_id": self.answer.doc_id,
            "start": self.answer.start,
            "end": self.answer.end,
        }


@dataclass(frozen=True)
class Corpus:
    """Documents and questions together, validated against each other on construction."""

    documents: tuple[Document, ...]
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        by_id: dict[str, Document] = {}
        for document in self.documents:
            if document.doc_id in by_id:
                raise CorpusError(f"duplicate doc_id {document.doc_id!r}")
            by_id[document.doc_id] = document
        seen: set[str] = set()
        for question in self.questions:
            if question.question_id in seen:
                raise CorpusError(f"duplicate question_id {question.question_id!r}")
            seen.add(question.question_id)
            document = by_id.get(question.answer.doc_id)
            if document is None:
                raise CorpusError(
                    f"question {question.question_id} annotates unknown document "
                    f"{question.answer.doc_id!r}"
                )
            if question.answer.end > document.length:
                raise CorpusError(
                    f"question {question.question_id} annotates characters "
                    f"{question.answer.start}:{question.answer.end} of a document only "
                    f"{document.length} characters long"
                )

    @property
    def by_id(self) -> dict[str, Document]:
        return {document.doc_id: document for document in self.documents}

    def answer_text(self, question: Question) -> str:
        document = self.by_id[question.answer.doc_id]
        return document.slice(question.answer.start, question.answer.end)

    @property
    def mean_answer_length(self) -> float:
        if not self.questions:
            return 0.0
        return round(
            sum(question.answer.length for question in self.questions) / len(self.questions), 1
        )


def write_corpus(path: Path, corpus: Corpus) -> int:
    """Write a corpus as a single JSON file and return the question count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "text": d.text} for d in corpus.documents
        ],
        "questions": [q.as_dict() for q in corpus.questions],
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return len(corpus.questions)


def load_corpus(path: Path) -> Corpus:
    """Read a corpus, failing on a malformed file rather than on a later confusing error."""
    if not path.exists():
        raise CorpusError(f"corpus not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path} is not valid JSON: {exc}") from exc
    for key in ("documents", "questions"):
        if key not in payload:
            raise CorpusError(f"{path} has no {key!r} key")
    documents = tuple(
        Document(doc_id=str(raw["doc_id"]), text=str(raw["text"]), title=str(raw.get("title", "")))
        for raw in payload["documents"]
    )
    questions = tuple(
        Question(
            question_id=str(raw["question_id"]),
            text=str(raw["text"]),
            answer=AnswerSpan(
                doc_id=str(raw["doc_id"]), start=int(raw["start"]), end=int(raw["end"])
            ),
        )
        for raw in payload["questions"]
    )
    return Corpus(documents=documents, questions=questions)


def total_characters(documents: Iterable[Document]) -> int:
    return sum(document.length for document in documents)
