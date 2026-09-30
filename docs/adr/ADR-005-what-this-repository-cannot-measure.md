# ADR-005: What this repository cannot measure, and what it would take

Status: accepted
Date: 2026-09-29

## Context

Every figure here comes from a seeded generator and a deterministic engine running offline.
That buys reproducibility, which is the right trade for an artefact a reviewer clones and runs
in two minutes. It does not buy realism, and the difference has to be written down where a
reader will find it rather than discovered by a reviewer who then discounts everything else.

## Decision

State the boundary claim by claim.

**Claims that survive any corpus.** That ground truth annotated on documents and retrieval over
chunks cannot be compared without losing information. That an overlap-scored recall can rise to
a perfect score while answerability does not move, because one is a function of k and the other
is not. That for a fixed-size chunker the severance rate under uniform placement is
`1 - min(1, (size - length + 1) / stride)`. These are logic and arithmetic.

**Claims that are about this corpus and nobody's pipeline.** Every rate. They demonstrate that
the phenomenon has a size, not that it has this size for you. The generator knobs are
documented so a reader can set their own answer lengths and re-measure.

**The residual is a finding, not an error.** The closed form lands **within 1.5 points** of
measurement on uniformly placed spans and is **out by 12.0 points** on the shipped corpus. Span
placement here is not uniform: answers sit at sentence boundaries inside paragraphs of similar
length, which correlates with the chunk grid. A model that matched real text perfectly would be
the suspicious result, and publishing the residual is what makes the rest of the model
believable.

**Sentence and paragraph chunkers have no closed form here.** Their boundaries are a property
of the text rather than of an arithmetic progression, so `CHUNKER_GEOMETRY` deliberately
excludes them and the comparison table reports measurements only. Applying the formula anyway
would produce a tidy number describing nothing.

**The taxonomy assumes the annotation is right.** Every class is defined against the annotated
span. A span annotated on the wrong sentence is classified confidently and wrongly, and nothing
here detects that. An annotation quality pass is a different tool.

**No generation step is measured.** The repository stops at retrieval. A chunk containing the
span is necessary for an answer and not sufficient, and whether a model can use it is out of
scope. Any reader tempted to read strict recall as an answer rate should not.

**The default retriever is lexical.** ADR-002 lists exactly which columns a model swap would
move and which it cannot.

## Alternatives considered

**Ship a corpus captured from a real system.** More convincing, not distributable, not
reproducible, impossible to vary. A seeded generator with its knobs exposed is worth more than
one anonymised snapshot.

**Quietly present the corpus rates as general.** The standard move, and the reason most
portfolio numbers are discounted on sight.

**Omit the section.** Leaves the reviewer to find the boundary themselves, at which point they
are auditing the author rather than the tool.

## Consequences

The README leads with what the tool cannot establish, which costs part of the first screen and
buys the only thing that makes the rest worth reading.

The future work has well defined jobs rather than a wish list: a generation step to test
sufficiency, and a dense retriever in CI to assert rather than argue the boundary this record
draws.
