"""The command line surface. Exit codes are an interface, documented in the README.

`evaluate` returns 3 when recall@k misattributes at least one question, which is the tool
working rather than the tool failing. A CI step that treats every non-zero code as a crash
reports a real finding as a broken tool, and the fix a team reaches for is to stop running it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rfa import exit_codes
from rfa.chunking import CHUNKERS, get_chunker
from rfa.corpus import load_corpus, write_corpus
from rfa.errors import ConfigError, CorpusError, RfaError
from rfa.evaluate import evaluate
from rfa.generator import GeneratorConfig, generate
from rfa.retrieve import pool_size

DEFAULT_CORPUS = Path("fixtures/policy_corpus.json")
DEFAULT_CHUNKER = "fixed-320"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfa",
        description=(
            "Attribute RAG retrieval failures to their actual cause, separating the ones a "
            "retrieval change can fix from the ones only re-chunking can."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write the deterministic policy corpus")
    gen.add_argument("--out", type=Path, default=DEFAULT_CORPUS)
    gen.add_argument("--documents", type=int, default=120)
    gen.add_argument("--questions", type=int, default=360)
    gen.add_argument("--seed", type=int, default=20260930)

    for name, help_text in (
        ("evaluate", "classify every question under one chunker"),
        ("report", "write a self-contained HTML report"),
        ("explain", "say why one question was classified as it was"),
    ):
        node = sub.add_parser(name, help=help_text)
        if name == "explain":
            node.add_argument("question_id")
        node.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
        node.add_argument("--chunker", default=DEFAULT_CHUNKER, choices=sorted(CHUNKERS))
        node.add_argument("-k", type=int, default=5)
        node.add_argument("--pool", type=int, default=None)
        if name == "evaluate":
            node.add_argument("--json", type=Path, default=None)
            node.add_argument("--require-questions", type=int, default=None)
        if name == "report":
            node.add_argument("--out", type=Path, default=Path("docs/report.html"))

    cmp_ = sub.add_parser("compare", help="evaluate every chunker and tabulate")
    cmp_.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    cmp_.add_argument("-k", type=int, default=5)
    cmp_.add_argument("--json", type=Path, default=None)

    swp = sub.add_parser("sweep", help="raise k and watch the two recalls diverge")
    swp.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    swp.add_argument("--chunker", default=DEFAULT_CHUNKER, choices=sorted(CHUNKERS))
    swp.add_argument("--max-k", type=int, default=50)
    return parser


def _cmd_generate(args: argparse.Namespace, out) -> int:
    corpus = generate(
        GeneratorConfig(documents=args.documents, questions=args.questions, seed=args.seed)
    )
    written = write_corpus(args.out, corpus)
    print(
        f"wrote {len(corpus.documents)} documents and {written} questions to {args.out}", file=out
    )
    return exit_codes.OK


def _evaluation(args: argparse.Namespace):
    corpus = load_corpus(args.corpus)
    return evaluate(corpus, get_chunker(args.chunker), args.chunker, k=args.k, pool=args.pool)


def _cmd_evaluate(args: argparse.Namespace, out) -> int:
    result = _evaluation(args)
    summary = result.summary()
    print(json.dumps(summary, indent=2), file=out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"summary": summary, "outcomes": [o.as_dict() for o in result.outcomes]}, indent=2
            ),
            encoding="utf-8",
        )
    if args.require_questions is not None and result.total < args.require_questions:
        print(
            f"the evaluation set holds {result.total} questions, below the "
            f"{args.require_questions} required",
            file=out,
        )
        return exit_codes.UNDERPOWERED
    misattributed = result.false_hits + result.severed_misses
    return exit_codes.FINDING if misattributed else exit_codes.OK


def _cmd_compare(args: argparse.Namespace, out) -> int:
    corpus = load_corpus(args.corpus)
    header = (
        f"{'chunker':<22}{'chunks':>8}{'answerable':>12}{'recall@k':>10}"
        f"{'strict':>9}{'false hits':>12}{'severed':>9}"
    )
    print(header, file=out)
    print("-" * len(header), file=out)
    rows = []
    for name in sorted(CHUNKERS):
        result = evaluate(corpus, get_chunker(name), name, k=args.k)
        row = result.summary()
        rows.append(row)
        print(
            f"{name:<22}{row['chunks']:>8}{row['answerable_pct']:>11}%{row['recall_generous_pct']:>9}%"
            f"{row['recall_strict_pct']:>8}%{row['false_hits']:>12}{row['severed_misses']:>9}",
            file=out,
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return exit_codes.FINDING if any(r["false_hits"] or r["severed_misses"] for r in rows) else 0


def _cmd_sweep(args: argparse.Namespace, out) -> int:
    corpus = load_corpus(args.corpus)
    chunker = get_chunker(args.chunker)
    print(f"{'k':>6}{'recall@k':>12}{'strict':>10}{'answerable':>13}", file=out)
    print("-" * 41, file=out)
    ks = [k for k in (1, 3, 5, 10, 20, 50, 100, 200) if k <= args.max_k] or [args.max_k]
    ceiling = None
    for k in ks:
        result = evaluate(corpus, chunker, args.chunker, k=k, pool=max(pool_size(k), k))
        summary = result.summary()
        ceiling = summary["answerable_pct"]
        print(
            f"{k:>6}{summary['recall_generous_pct']:>11}%{summary['recall_strict_pct']:>9}%"
            f"{summary['answerable_pct']:>12}%",
            file=out,
        )
    print(f"\nthe answerable ceiling does not move with k: {ceiling}%", file=out)
    return exit_codes.OK


def _cmd_explain(args: argparse.Namespace, out) -> int:
    result = _evaluation(args)
    matches = [o for o in result.outcomes if o.question_id == args.question_id]
    if not matches:
        raise ConfigError(f"question {args.question_id!r} is not in {args.corpus}")
    outcome = matches[0]
    print(f"{outcome.question_id}: {outcome.failure}", file=out)
    print(f"  answer length          {outcome.answer_length} characters", file=out)
    print(f"  chunks overlapping it  {outcome.overlapping_chunks}", file=out)
    print(f"  chunks containing it   {outcome.containing_chunks}", file=out)
    print(f"  best overlapping rank  {outcome.best_overlapping_rank}", file=out)
    print(f"  best containing rank   {outcome.best_containing_rank}", file=out)
    print(f"  scored a hit at k={result.k}?  {outcome.scored_hit_generous}", file=out)
    print(f"  actually answerable?   {outcome.answerable}", file=out)
    return exit_codes.FINDING if outcome.misattributed else exit_codes.OK


def _cmd_report(args: argparse.Namespace, out) -> int:
    from rfa.report import build_report

    result = _evaluation(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(result), encoding="utf-8")
    print(f"wrote {args.out} for {result.total} questions", file=out)
    return exit_codes.FINDING if result.false_hits + result.severed_misses else exit_codes.OK


HANDLERS = {
    "generate": _cmd_generate,
    "evaluate": _cmd_evaluate,
    "compare": _cmd_compare,
    "sweep": _cmd_sweep,
    "explain": _cmd_explain,
    "report": _cmd_report,
}


def main(argv: Sequence[str] | None = None, out=None) -> int:
    """Entry point. Returns a code rather than calling sys.exit, so tests can drive it."""
    out = out or sys.stdout
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return HANDLERS[args.command](args, out)
    except (ConfigError, CorpusError) as exc:
        print(f"error: {exc}", file=out)
        return exit_codes.USAGE
    except RfaError as exc:  # pragma: no cover - defensive, no current path reaches it
        print(f"error: {exc}", file=out)
        return exit_codes.USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
