# ADR-003: A wider candidate pool separates two retrieval failures

Status: accepted
Date: 2026-09-27

## Context

When a containing chunk exists and did not make the top k, the taxonomy has to say something
more useful than "retrieval missed it", because two very different situations produce that
outcome and they have different price tags.

In the first, the chunk is there at rank eleven. A reranker over a wider candidate set fixes
it, and that is a cheap, well understood purchase.

In the second, the chunk is nowhere near the top of the ranking at all. No reranker helps,
because a reranker only reorders what it is given. That needs a different embedding model,
query rewriting, or a hybrid lexical pass, all of which are larger projects.

Collapsing the two is how a retrieval roadmap becomes a matter of taste.

## Decision

Rank against a candidate pool four times wider than k, and split the failures on it.
`rank_miss` means the containing chunk was inside the pool and below k. `embedding_miss` means
it did not reach the pool at all.

Four is the usual reranker fan-out, and it is a parameter rather than a constant in the code so
a team whose reranker sees a hundred candidates can say so.

## Alternatives considered

**Rank everything and report the raw rank.** More information and no decision attached. A rank
of 340 out of 769 means nothing without a threshold, so the reader invents one.

**Two separate k values and two separate runs.** The same information at twice the compute and
with an extra concept for the reader to hold.

**A score threshold instead of a rank threshold.** Similarity scores are not comparable across
retrievers or corpora, so a threshold tuned on one is meaningless on another. Rank is at least
ordinal and at least stable.

## Consequences

The taxonomy can name a purchase for each class rather than a team, which is the difference
between a report that gets read and one that gets forwarded.

The pool costs compute: every question is ranked against the whole index and then truncated.
The inverted index makes that affordable, and the cost is documented rather than hidden.

There is a boundary here. The split is about this retriever's ranking, so like the other rank
figures it moves with the model. ADR-002 lists which numbers do and which do not.
