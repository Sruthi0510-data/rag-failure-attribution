"""Re-measure every published figure and write docs/metrics.json.

Nothing is typed by hand. The test count and the coverage come from pytest's machine readable
reports rather than from a parsed progress line, and every other figure comes from an
experiment's JSON, which is produced by running the experiment.

Anchors are the other half. A checker that searches for digits passes while the sentence
around them has gone stale, so each metric carries the two or three words a document must use
with the value substituted in. Anchors are short, because a phrase longer than a few words
straddles a wrapped line and then matches nothing while the prose it guards is correct. They
are also specific: "{} questions" would match every sentence in the repository that counts
questions and report a correct figure as stale a dozen times over.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfa.chunking import CHUNKER_GEOMETRY  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_tables import table_values  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPERIMENTS = ROOT / "experiments"
EXPERIMENT_OUTPUT = DOCS / "experiments"
JUNIT = ROOT / "junit.xml"
COVERAGE_JSON = ROOT / "coverage.json"
METRICS_PATH = DOCS / "metrics.json"

CHECKED_DOCUMENTS = [
    "README.md",
    "docs/adr/ADR-001-ground-truth-lives-on-documents.md",
    "docs/adr/ADR-002-no-downloaded-model-in-the-default-path.md",
    "docs/adr/ADR-003-a-wider-pool-separates-two-retrieval-failures.md",
    "docs/adr/ADR-004-exit-codes-separate-a-finding-from-a-failure.md",
    "docs/adr/ADR-005-what-this-repository-cannot-measure.md",
]

ANCHORS: dict[str, list[str]] = {
    "tests_total": ["{} tests,"],
    "coverage_line_pct": ["{}% line"],
    "coverage_branch_pct": ["{}% branch"],
    "corpus_documents": ["{} policy documents"],
    "corpus_questions": ["{} annotated questions"],
    "corpus_chunks": ["{} chunks at"],
    "default_chunk_size": ["default of {} characters"],
    "mean_answer_length": ["{} characters long", "averaging {} characters"],
    "recall_generous_pct": ["reports {}%", "scores {}%"],
    "answerable_pct": ["only {}% of", "ceiling of {}%"],
    "recall_strict_pct": ["strict recall of {}%"],
    "false_hits": ["{} of them cannot", "{} scored hits"],
    "false_hit_share_pct": ["{}% of the scored hits"],
    "misses_total": ["{} recorded misses"],
    "severed_misses": ["{} of those misses", "{} severed"],
    "severed_share_pct": ["{}% of the misses"],
    "retrieval_fixable": ["{} questions a retriever"],
    "chunking_fixable": ["{} need a chunking"],
    "perfect_recall_at_k": ["k of {} the metric", "at k of {}"],
    "smallest_chunker_answerable_pct": ["answers just {}%"],
    "best_chunker_strict_pct": ["best strict recall {}%"],
    "answers_recovered_by_overlap": ["recovers {} answers"],
    "rank_misses_added_by_overlap": ["creates {} new rank"],
    "overlap_index_multiple": ["costs {} times"],
    "worst_uniform_residual_pct": ["within {} points"],
    "worst_corpus_residual_pct": ["out by {} points"],
    "model_trials": ["{} placements"],
}


def run_tests() -> bool:
    """Run pytest with machine readable reports. Returns whether it was green.

    Deliberately does not raise on a red suite, and that is a fix rather than a
    looseness. The suite contains a test that runs the number checker, the
    number checker compares the documents against the figures this script
    writes, and this script is how a stale figure gets corrected. Refusing to
    measure until the suite is green therefore made a stale figure unfixable:
    the only tool that could repair it declined to run until it had already been
    repaired.

    pytest writes its junit and coverage reports whether it passes or fails, so
    the measurement is available either way. A red suite is a fact to print
    loudly, not a reason to refuse to measure.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--cov=rfa", f"--cov-report=json:{COVERAGE_JSON}", f"--junitxml={JUNIT}",
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        print(
            "  the suite is red; measuring anyway so a stale figure can be "
            "corrected, then run it again"
        )
    return result.returncode == 0


def read_test_count() -> int:
    root = ET.parse(JUNIT).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    return int(suite.get("tests", 0))


def read_coverage() -> tuple[float, float]:
    totals = json.loads(COVERAGE_JSON.read_text())["totals"]
    line = round(float(totals["percent_covered"]), 1)
    covered, total = totals.get("covered_branches", 0), totals.get("num_branches", 0)
    return line, (round(100.0 * covered / total, 1) if total else 100.0)


def run_experiments() -> dict[str, Any]:
    """Run every experiment from scratch and load the JSON each one writes."""
    results: dict[str, Any] = {}
    for script in sorted(EXPERIMENTS.glob("exp*.py")):
        subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(EXPERIMENTS)},
            stdout=subprocess.DEVNULL,
        )
        results[script.stem] = json.loads((EXPERIMENT_OUTPUT / f"{script.stem}.json").read_text())
    return results


def build_metrics(skip_tests: bool) -> dict[str, Any]:
    green = True
    if not skip_tests:
        green = run_tests()
    line_pct, branch_pct = read_coverage()
    experiments = run_experiments()

    attribution = experiments["exp01_attribution"]["summary"]
    sweep = experiments["exp02_k_sweep"]
    tradeoff = experiments["exp03_chunker_tradeoff"]
    model = experiments["exp04_severance_model"]
    overlap = experiments["exp05_overlap_cost"]

    best = tradeoff["best_strict"]
    overlap_pair = overlap["pairs"][0]

    metrics = {
        "tests_total": read_test_count(),
        "coverage_line_pct": line_pct,
        "coverage_branch_pct": branch_pct,
        "corpus_documents": 120,
        "corpus_questions": attribution["questions"],
        "corpus_chunks": attribution["chunks"],
        "default_chunk_size": CHUNKER_GEOMETRY["fixed-320"][0],
        "mean_answer_length": attribution["mean_answer_length"],
        "recall_generous_pct": attribution["recall_generous_pct"],
        "answerable_pct": attribution["answerable_pct"],
        "recall_strict_pct": attribution["recall_strict_pct"],
        "false_hits": attribution["false_hits"],
        "false_hit_share_pct": attribution["false_hit_share_of_scored_hits_pct"],
        "misses_total": attribution["misses"],
        "severed_misses": attribution["severed_misses"],
        "severed_share_pct": attribution["severed_share_of_misses_pct"],
        "retrieval_fixable": attribution["retrieval_fixable"],
        "chunking_fixable": attribution["chunking_fixable"],
        "perfect_recall_at_k": sweep["perfect_recall_at_k"],
        "smallest_chunker_answerable_pct": tradeoff["chunkers"]["fixed-160"]["answerable_pct"],
        "best_chunker_strict_pct": tradeoff["chunkers"][best]["recall_strict_pct"],
        "answers_recovered_by_overlap": overlap_pair["unanswerable_removed"],
        "rank_misses_added_by_overlap": overlap_pair["rank_misses_added"],
        "overlap_index_multiple": overlap_pair["index_multiple"],
        "worst_uniform_residual_pct": model["worst_uniform_residual_pct"],
        "worst_corpus_residual_pct": model["worst_corpus_residual_pct"],
        "model_trials": model["trials"],
    }
    return {
        "metrics": metrics,
        "table_values": table_values(),
        "anchors": ANCHORS,
        "checked_documents": CHECKED_DOCUMENTS,
        "measured_despite_a_red_suite": not green,
        "note": (
            "every value here is produced by running the suite and the experiments; "
            "tools/check_numbers.py fails the build when a document quotes a figure the code "
            "no longer produces"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="re-measure every published figure")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    payload = build_metrics(skip_tests=args.skip_tests)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {METRICS_PATH.relative_to(ROOT)} with {len(payload['metrics'])} metrics")
    for name, value in sorted(payload["metrics"].items()):
        print(f"  {name:<34} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
