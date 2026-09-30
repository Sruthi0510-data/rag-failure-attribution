"""The command line surface, with the exit codes treated as the interface they are."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from rfa import exit_codes
from rfa.chunking import get_chunker
from rfa.cli import main
from rfa.corpus import load_corpus
from rfa.evaluate import evaluate
from tests.conftest import CORPUS_PATH


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def outcomes():
    result = evaluate(load_corpus(CORPUS_PATH), get_chunker("fixed-320"), "fixed-320", k=5)
    return result.outcomes


def test_generate_writes_a_corpus_and_reports_the_counts(tmp_path: Path):
    target = tmp_path / "c.json"
    code, text = run("generate", "--out", str(target), "--documents", "6", "--questions", "12")
    assert code == exit_codes.OK
    assert target.exists()
    assert "6 documents" in text and "12 questions" in text


def test_generate_is_reproducible_from_the_command_line(tmp_path: Path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for target in (first, second):
        run("generate", "--out", str(target), "--documents", "5", "--questions", "5", "--seed", "3")
    assert first.read_text() == second.read_text()


def test_evaluate_returns_the_finding_code_when_recall_misattributes():
    code, text = run("evaluate", "--corpus", str(CORPUS_PATH), "--chunker", "fixed-320")
    assert code == exit_codes.FINDING
    assert json.loads(text)["false_hits"] > 0


def test_evaluate_returns_ok_for_a_chunker_that_severs_nothing():
    code, text = run("evaluate", "--corpus", str(CORPUS_PATH), "--chunker", "paragraph")
    summary = json.loads(text)
    assert summary["false_hits"] == 0
    assert summary["severed_misses"] == 0
    assert code == exit_codes.OK


def test_a_finding_and_a_failure_to_answer_have_different_exit_codes(tmp_path: Path):
    """A CI step treating every non-zero code as a crash would report a finding as a crash."""
    finding, _ = run("evaluate", "--corpus", str(CORPUS_PATH))
    failure, _ = run("evaluate", "--corpus", str(tmp_path / "absent.json"))
    assert finding == exit_codes.FINDING
    assert failure == exit_codes.USAGE
    assert finding != failure


def test_evaluate_reports_a_missing_corpus_by_path(tmp_path: Path):
    code, text = run("evaluate", "--corpus", str(tmp_path / "absent.json"))
    assert code == exit_codes.USAGE
    assert "corpus not found" in text


def test_evaluate_writes_every_outcome_to_json_when_asked(tmp_path: Path):
    target = tmp_path / "nested" / "eval.json"
    run("evaluate", "--corpus", str(CORPUS_PATH), "--json", str(target))
    payload = json.loads(target.read_text())
    assert payload["summary"]["questions"] == len(payload["outcomes"])


def test_requiring_more_questions_than_the_set_holds_returns_underpowered():
    code, text = run("evaluate", "--corpus", str(CORPUS_PATH), "--require-questions", "100000")
    assert code == exit_codes.UNDERPOWERED
    assert "below the" in text


def test_requiring_a_satisfiable_question_count_does_not_change_the_verdict():
    code, _ = run("evaluate", "--corpus", str(CORPUS_PATH), "--require-questions", "10")
    assert code == exit_codes.FINDING


def test_the_underpowered_check_is_opt_in():
    """A check wired to fail by default gets weakened within a week."""
    code, _ = run("evaluate", "--corpus", str(CORPUS_PATH))
    assert code == exit_codes.FINDING


def test_evaluate_accepts_an_explicit_k_and_pool():
    _, text = run("evaluate", "--corpus", str(CORPUS_PATH), "-k", "3", "--pool", "30")
    summary = json.loads(text)
    assert summary["k"] == 3
    assert summary["pool"] == 30


def test_a_pool_smaller_than_k_is_a_usage_error():
    code, text = run("evaluate", "--corpus", str(CORPUS_PATH), "-k", "10", "--pool", "2")
    assert code == exit_codes.USAGE
    assert "cannot contain the top" in text


def test_compare_tabulates_every_chunker():
    code, text = run("compare", "--corpus", str(CORPUS_PATH))
    assert code == exit_codes.FINDING
    for name in ("fixed-160", "fixed-320", "paragraph", "sentence-3"):
        assert name in text


def test_compare_writes_its_rows_to_json(tmp_path: Path):
    target = tmp_path / "compare.json"
    run("compare", "--corpus", str(CORPUS_PATH), "--json", str(target))
    rows = json.loads(target.read_text())["rows"]
    assert len(rows) >= 6


def test_the_sweep_shows_the_ceiling_standing_still():
    code, text = run("sweep", "--corpus", str(CORPUS_PATH), "--max-k", "20")
    assert code == exit_codes.OK
    assert "does not move with k" in text


def test_the_sweep_honours_a_small_max_k():
    _, text = run("sweep", "--corpus", str(CORPUS_PATH), "--max-k", "1")
    assert "1" in text


def test_explain_names_the_class_and_the_evidence():
    code, text = run("explain", "q-0000", "--corpus", str(CORPUS_PATH))
    assert code in (exit_codes.OK, exit_codes.FINDING)
    assert "answer length" in text
    assert "actually answerable" in text


def test_explain_on_an_unknown_question_says_so():
    code, text = run("explain", "q-99999", "--corpus", str(CORPUS_PATH))
    assert code == exit_codes.USAGE
    assert "is not in" in text


def test_explain_returns_the_finding_code_for_a_misattributed_question():
    misattributed = next(o for o in outcomes() if o.misattributed)
    code, _ = run("explain", misattributed.question_id, "--corpus", str(CORPUS_PATH))
    assert code == exit_codes.FINDING


def test_explain_returns_ok_for_a_correctly_attributed_question():
    clean = next(o for o in outcomes() if not o.misattributed)
    code, _ = run("explain", clean.question_id, "--corpus", str(CORPUS_PATH))
    assert code == exit_codes.OK


def test_report_writes_a_self_contained_html_file(tmp_path: Path):
    target = tmp_path / "nested" / "report.html"
    code, text = run("report", "--corpus", str(CORPUS_PATH), "--out", str(target))
    assert code == exit_codes.FINDING
    assert "360 questions" in text
    body = target.read_text()
    assert body.startswith("<!doctype html>")
    assert "<link" not in body


def test_report_returns_ok_for_a_chunker_that_misattributes_nothing(tmp_path: Path):
    code, _ = run(
        "report", "--corpus", str(CORPUS_PATH), "--chunker", "paragraph",
        "--out", str(tmp_path / "r.html"),
    )
    assert code == exit_codes.OK


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        main([])


def test_an_unknown_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
