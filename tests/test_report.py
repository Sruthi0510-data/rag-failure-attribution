"""The HTML report, including the class collision a passing unit suite cannot see."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rfa.chunking import get_chunker
from rfa.evaluate import evaluate
from rfa.failures import FailureClass
from rfa.report import CLASSES, OWNER_LABEL, STYLESHEET, build_report


@pytest.fixture(scope="module")
def finding_report(shipped_corpus) -> str:
    return build_report(evaluate(shipped_corpus, get_chunker("fixed-320"), "fixed-320", k=5))


@pytest.fixture(scope="module")
def clean_report(shipped_corpus) -> str:
    return build_report(evaluate(shipped_corpus, get_chunker("paragraph"), "paragraph", k=5))


def _classes_used(html: str) -> set[str]:
    used: set[str] = set()
    for match in re.finditer(r"class=['\"]([^'\"]+)['\"]", html):
        used.update(match.group(1).split())
    return used


def test_the_report_is_a_complete_html_document(finding_report):
    assert finding_report.startswith("<!doctype html>")
    assert finding_report.rstrip().endswith("</html>")


def test_the_report_inlines_its_stylesheet(finding_report):
    assert "<style>" in finding_report
    assert "<link" not in finding_report


def test_a_finding_report_shows_the_finding_verdict(finding_report):
    assert 'class="verdict-finding"' in finding_report
    assert "MISATTRIBUTES" in finding_report


def test_a_clean_report_shows_the_clear_verdict(clean_report):
    assert 'class="verdict-clear"' in clean_report
    assert "ATTRIBUTES EVERY FAILURE CORRECTLY" in clean_report


def test_the_two_verdicts_never_appear_together(finding_report, clean_report):
    """Matched on the class attribute, not the bare name, since both rules live in the CSS."""
    assert 'class="verdict-clear"' not in finding_report
    assert 'class="verdict-finding"' not in clean_report


@pytest.mark.parametrize("heading", ["summary", "taxonomy", "index"])
def test_every_heading_the_screenshot_tool_frames_by_is_present(heading, finding_report):
    assert f'id="{heading}"' in finding_report


def test_the_report_names_the_chunker_and_k(finding_report):
    assert "fixed-320" in finding_report
    assert "k of 5" in finding_report


def test_the_taxonomy_names_an_owner_for_every_class_it_shows(finding_report):
    assert "whoever chose the chunk size" in finding_report
    assert "whoever owns the reranker" in finding_report


def test_every_failure_class_has_an_owner_label():
    """A class with no owner would print an empty cell and defeat the point of the table."""
    assert set(OWNER_LABEL) == set(FailureClass)


def test_the_bars_are_sized_from_the_counts_rather_than_hardcoded(finding_report):
    widths = {int(m) for m in re.findall(r"bar-fill\" style=\"width:(\d+)px", finding_report)}
    assert len(widths) > 1


def test_a_clean_report_still_shows_the_hit_row(clean_report):
    assert "hit" in clean_report


def test_every_class_in_the_markup_is_declared_and_styled(finding_report, clean_report):
    """Pins the collision that renders a headline in its own background colour."""
    used = _classes_used(finding_report) | _classes_used(clean_report)
    assert sorted(used - CLASSES) == []
    assert sorted(name for name in used if f".{name}" not in STYLESHEET) == []


def test_every_declared_class_has_a_rule_so_the_list_cannot_rot():
    assert sorted(name for name in CLASSES if f".{name}" not in STYLESHEET) == []


def test_the_two_verdict_classes_have_different_backgrounds():
    finding = re.search(r"\.verdict-finding \{[^}]*background: (#[0-9a-f]+)", STYLESHEET)
    clear = re.search(r"\.verdict-clear \{[^}]*background: (#[0-9a-f]+)", STYLESHEET)
    assert finding and clear
    assert finding.group(1) != clear.group(1)


def test_the_two_owner_colours_are_distinguishable():
    """Chunking work and retrieval work must not read as the same colour in the table."""
    chunking = re.search(r"\.owner-chunking \{ color: (#[0-9a-f]+)", STYLESHEET)
    retrieval = re.search(r"\.owner-retrieval \{ color: (#[0-9a-f]+)", STYLESHEET)
    assert chunking and retrieval
    assert chunking.group(1) != retrieval.group(1)


def test_chunker_names_are_escaped_rather_than_interpolated_raw(shipped_corpus):
    result = evaluate(shipped_corpus, get_chunker("fixed-320"), "<script>x</script>", k=5)
    html = build_report(result)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


ROOT = Path(__file__).resolve().parents[1]


def test_the_proof_of_work_transcript_exists_and_covers_several_exit_codes():
    """A transcript that only ever shows a zero proves the least interesting thing."""
    payload = json.loads((ROOT / "docs" / "proof-of-work.json").read_text())
    codes = {capture["exit_code"] for capture in payload["captures"]}
    assert len(codes) >= 3


def test_every_captured_command_exited_as_the_readme_claims():
    payload = json.loads((ROOT / "docs" / "proof-of-work.json").read_text())
    for capture in payload["captures"]:
        assert capture["exit_code"] == capture["expected_exit_code"], capture["command"]


def test_every_captured_command_carries_a_hash_of_its_untruncated_output():
    payload = json.loads((ROOT / "docs" / "proof-of-work.json").read_text())
    for capture in payload["captures"]:
        assert len(capture["output_sha256"]) == 64


def test_the_transcript_leaks_no_absolute_path():
    svg = (ROOT / "docs" / "img" / "proof-of-work.svg").read_text(encoding="utf-8")
    assert str(ROOT) not in svg
    assert "/usr/bin/python" not in svg


def test_the_transcript_is_valid_svg_with_a_title_and_description():
    import xml.etree.ElementTree as ET

    namespace = "{http://www.w3.org/2000/svg}"
    root = ET.parse(ROOT / "docs" / "img" / "proof-of-work.svg").getroot()
    assert root.find(f"{namespace}title") is not None
    assert root.find(f"{namespace}desc") is not None


def test_the_readme_embeds_the_transcript():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/img/proof-of-work.svg" in text
