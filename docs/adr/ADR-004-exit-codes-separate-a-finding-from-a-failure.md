# ADR-004: Exit codes separate a finding from a failure

Status: accepted
Date: 2026-09-28

## Context

An auditing tool has three genuinely different outcomes and the Unix convention has room for
two. "I ran and found nothing", "I ran and found something" and "I could not run" collapse into
zero or non-zero, and the collapse has a predictable consequence: a CI step that treats every
non-zero code as a crash reports a real finding as a broken tool, somebody marks the step
`continue-on-error`, and nothing is checked from then on.

There is a fourth outcome specific to an evaluation tool. A run can complete, report a clean
result, and rest on an evaluation set too small for that result to mean anything.

## Decision

Four codes, documented in the README as an interface.

| code | meaning |
| --- | --- |
| 0 | ran, recall@k attributed every failure correctly |
| 3 | ran, found misattributed questions. The tool worked |
| 2 | could not answer: bad arguments, unreadable corpus, an annotation off its document |
| 4 | ran, and the evaluation set is smaller than the caller required |

Three rather than one, deliberately. One is what a crashed interpreter returns, and the gap
makes the distinction visible in a CI log rather than merely documented.

Code 4 is opt in, behind `--require-questions`. A check wired to fail by default gets weakened
within a week and then nothing is checked, which is the same failure arriving by another road.

## Alternatives considered

**Zero for everything, findings on stdout.** Forces every caller to parse output to learn
whether to act, and the parser is what breaks when the output format improves.

**Non-zero for any finding, code 1.** The common choice and the one that produces the
`continue-on-error` outcome, because 1 is indistinguishable from a traceback.

**A minimum question count enforced always.** Rejected for the same reason as a failing check:
a tool that refuses to run on a small evaluation set is a tool that gets run on a fork with the
check removed.

## Consequences

A CI step reads `if [ $code -eq 3 ]` and knows a finding from a crash without parsing anything.

The codes are a compatibility surface now. The constants live in one module with the meanings
in their docstrings, and tests assert that a finding and a failure to answer return different
codes rather than asserting the literals alone, so the property survives a renumbering.
