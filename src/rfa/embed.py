"""A deterministic lexical embedder, and the reason the default path does not download a model.

The finding this repository publishes is that a share of retrieval failures are severance, and
severance is a property of the chunker and the annotation. No embedder appears anywhere in that
statement. Whether a chunk boundary falls inside an answer has nothing to do with how the chunk
is vectorised, so swapping in a transformer moves the rank-related classes around and cannot
move the severed ones at all. ADR-002 states that boundary and the experiments hold it up.

What the default does need is a retriever that genuinely fails sometimes, or the rank classes
would be empty and the taxonomy would be untested. TF-IDF cosine over the corpus vocabulary is
real retrieval with real weaknesses: it misses a paraphrase that shares no content words, which
is exactly the failure a dense model is bought to fix, and it produces both rank misses and
embedding misses on the shipped corpus.

The optional `transformers` extra swaps the vectoriser. The import sits below the input
validation, and a test makes the module unimportable to pin that ordering, because a machine
with the extra installed can never notice the bug.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rfa.errors import ConfigError

TOKEN = re.compile(r"[a-z0-9]+")
TRANSFORMERS_HINT = (
    "install the optional extra: pip install 'rag-failure-attribution[transformers]'"
)


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Deliberately boring and deliberately stable."""
    return TOKEN.findall(text.lower())


@dataclass
class TfidfEmbedder:
    """A vocabulary, its inverse document frequencies, and L2 normalised sparse vectors.

    Sparse dictionaries rather than dense arrays: the corpus vocabulary runs to a few thousand
    terms and a dense matrix would be almost entirely zeros, which costs memory and buys no
    clarity. The similarity loop iterates the shorter side, so a query of a dozen terms costs a
    dozen lookups per chunk rather than a walk over the vocabulary.
    """

    idf: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, documents: Sequence[str]) -> TfidfEmbedder:
        """Learn the vocabulary and its inverse document frequencies."""
        if not documents:
            raise ConfigError("cannot fit an embedder on an empty collection")
        frequency: dict[str, int] = {}
        for text in documents:
            for term in set(tokenise(text)):
                frequency[term] = frequency.get(term, 0) + 1
        total = len(documents)
        self.idf = {
            term: math.log((total + 1) / (count + 1)) + 1.0 for term, count in frequency.items()
        }
        self.fitted = True
        return self

    def encode(self, text: str) -> dict[str, float]:
        """Vectorise one string. Unknown terms are dropped rather than given a default weight.

        Giving an unseen term a weight would let a query term that appears nowhere in the index
        influence the ranking, which produces a similarity that is not a similarity to anything.
        """
        if not self.fitted:
            raise ConfigError("embedder must be fitted before encoding")
        counts: dict[str, float] = {}
        for term in tokenise(text):
            if term in self.idf:
                counts[term] = counts.get(term, 0.0) + 1.0
        if not counts:
            return {}
        vector = {term: (1.0 + math.log(count)) * self.idf[term] for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0.0:  # pragma: no cover - unreachable while idf is strictly positive
            return {}
        return {term: value / norm for term, value in vector.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Dot product of two L2 normalised sparse vectors, iterating the shorter side."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def transformer_embedder(model_name: str = "all-MiniLM-L6-v2"):  # pragma: no cover - needs extra
    """Load a sentence-transformers model, validating the argument before importing anything.

    Ordering matters: a caller who passes an empty model name should be told that immediately
    rather than after a multi-hundred-megabyte import, or worse, be told to install a package
    they do not need.
    """
    if not model_name or not model_name.strip():
        raise ConfigError("a transformer embedder needs a model name")
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ConfigError(f"the transformer path needs its extra; {TRANSFORMERS_HINT}") from exc
    return SentenceTransformer(model_name)
