# ADR-001: Ground truth lives on documents, and everything follows from that

Status: accepted
Date: 2026-09-24

## Context

An evaluation set for retrieval is built by a person reading a document and highlighting the
text that answers a question. The artefact is a character range in a document. It is produced
months before anyone picks a chunk size, by somebody who is not thinking about chunk sizes at
all, and it does not change when the chunk size does.

Retrieval returns chunks. So every retrieval metric has to compare a span to a set of chunk
ids, and the only comparison available is overlap. That comparison is the whole problem: a
chunk holding half an answer overlaps the span exactly as a chunk holding all of it does.

## Decision

Model the ground truth as it actually exists, a `(doc_id, start, end)` range, and give the span
two separate predicates rather than one.

`overlaps` asks whether a chunk shares any character with the answer. That is what an
evaluation harness can compute and what recall@k effectively uses.

`contained_by` asks whether a chunk holds every character of the answer. That is what the
pipeline needs to be able to answer the question, and no standard metric computes it.

Three numbers then come out of the same run: answerability, which asks whether any chunk in
the whole index contains the span; generous recall@k, built on `overlaps`; and strict recall@k,
built on `contained_by`. They are never collapsed.

## Alternatives considered

**Annotate on chunks.** Tidy, and it makes the ground truth a function of the chunk size, so
changing the chunker invalidates the evaluation set. Teams that do this cannot compare two
chunkers at all, which is the comparison they most need.

**Score overlap as a fraction and set a threshold.** A single number again, with a threshold
nobody can justify. Half an answer scores 0.5 and answers nothing, so the metric is not linear
in the thing it claims to measure.

**Require the retrieved chunk to be the one the annotator would have picked.** Assumes the
annotator picked a chunk, which returns to the first alternative.

## Consequences

The taxonomy in `failures.py` becomes possible, and with it the `false_hit` class: a question
the usual metric scores as a win while no chunk in the index contains the whole answer. That
class has no name in any retrieval evaluation the author has seen, and on the shipped corpus it
is the largest class of all.

The cost is that the evaluation set has to carry offsets. Question and document id are not
enough. Any team whose ground truth is only a document id can compute generous recall and
nothing else, and would have to re-annotate to use this tool. That is a real adoption cost and
it is the honest price of the distinction.
