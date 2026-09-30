# ADR-002: No downloaded model in the default path

Status: accepted
Date: 2026-09-26

## Context

This is a repository about retrieval quality, so the obvious default retriever is a sentence
transformer. It is what a reader expects and what a production system uses.

It also costs a model download between a reviewer and the first number, a non-deterministic
result across library versions, and a corpus of embeddings too large to commit. A reviewer who
cannot reproduce a headline figure in two minutes does not reproduce it at all.

## Decision

The default retriever is TF-IDF cosine over an inverted index built from the corpus itself.
Deterministic, offline, and with real weaknesses: it misses a paraphrase that shares no content
words, which is exactly the failure a dense model is bought to fix, and it produces both rank
misses and embedding misses on the shipped corpus so the taxonomy is exercised rather than
asserted.

The dense path exists behind an optional extra and is not what the README quotes.

The decision rests on a claim that has to be stated precisely, because the whole argument for
an offline default depends on it. **Severance is a property of the chunker and the annotation,
and contains no reference to an embedder.** Whether a chunk boundary falls inside an answer
does not depend on how the chunk is vectorised. So:

- **Survives a model swap unchanged**: answerability, the severed count, the false hit count,
  the entire chunker comparison's answerable column, and the closed form.
- **Moves with the model**: the rank miss and embedding miss counts, generous and strict
  recall at a given k, and therefore the specific k at which the metric reaches a perfect
  score.

## Alternatives considered

**A dense model as the default, with committed embeddings.** Reproducible and large, and the
embeddings would be a binary blob nobody can review. It also fixes the model version into the
repository forever.

**A dense model downloaded on first run.** Breaks the offline promise, and a CI job that pulls
a model from a hub is a job that fails on the day the hub is slow.

**A random-projection embedder.** Deterministic and fake. It would produce rank misses that
correspond to nothing, and the taxonomy's retrieval classes would be noise dressed as findings.

## Consequences

A reviewer clones and reproduces every number with one command and no network, which was worth
more than realism for the claims this repository makes.

The rank and embedding columns are about a lexical retriever and the README says so rather than
implying they generalise. A reader who wants those numbers for their own model has a documented
extra and a stated expectation about which columns should move.

The import of `sentence_transformers` sits below the argument validation in `transformer_embedder`,
and a test makes the module unimportable to pin that ordering, because a machine with the extra
installed can never notice the bug.
