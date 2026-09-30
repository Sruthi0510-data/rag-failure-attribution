"""A self-contained HTML report. Stable heading ids, and one class list the tests can check.

The capture tool frames screenshots by scrolling to a named h2 and clipping to the next, so a
renamed heading must break the capture loudly rather than silently produce half a table. Every
class used in the markup is declared in CLASSES and has a rule in STYLESHEET, and a test
asserts the two agree, because a verdict badge and a table cell colliding at equal specificity
renders the headline in its own background colour and no unit test can see that.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from rfa.failures import FailureClass

if TYPE_CHECKING:  # pragma: no cover
    from rfa.evaluate import Evaluation

CLASSES = frozenset(
    {
        "page", "masthead", "subtitle", "verdict-finding", "verdict-clear", "section",
        "grid", "cell", "cell-label", "cell-value", "cell-note", "data-table", "numeric",
        "owner-chunking", "owner-retrieval", "owner-none", "footnote", "bar", "bar-fill",
    }
)

STYLESHEET = """
:root { color-scheme: light; }
body { margin: 0; background: #f6f7f6; color: #1f2328;
       font-family: -apple-system, Helvetica, Arial, sans-serif; }
.page { max-width: 880px; margin: 0 auto; padding: 32px 24px 64px; }
.masthead { font-size: 26px; font-weight: 700; margin: 0 0 6px; }
.subtitle { color: #57606a; font-size: 14px; margin: 0 0 20px; }
.verdict-finding { display: inline-block; padding: 8px 16px; border-radius: 999px;
                   background: #8a3b12; color: #ffffff; font-weight: 700; font-size: 15px; }
.verdict-clear { display: inline-block; padding: 8px 16px; border-radius: 999px;
                 background: #0b6b5a; color: #ffffff; font-weight: 700; font-size: 15px; }
.section { margin-top: 36px; }
.section h2 { font-size: 18px; margin: 0 0 12px; padding-bottom: 6px;
              border-bottom: 1px solid #d8dee4; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.cell { background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px 16px; }
.cell-label { color: #57606a; font-size: 12px; text-transform: uppercase;
              letter-spacing: 0.04em; }
.cell-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
.cell-note { color: #57606a; font-size: 12px; margin-top: 4px; }
.data-table { width: 100%; border-collapse: collapse; background: #ffffff;
              border: 1px solid #d8dee4; font-size: 14px; }
.data-table th { text-align: left; padding: 10px 12px; border-bottom: 1px solid #d8dee4;
                 color: #57606a; font-weight: 600; font-size: 12px; text-transform: uppercase; }
.data-table td { padding: 10px 12px; border-bottom: 1px solid #eaeef2; }
.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.owner-chunking { color: #8a3b12; font-weight: 700; }
.owner-retrieval { color: #0b4f8a; font-weight: 700; }
.owner-none { color: #57606a; }
.bar { background: #eaeef2; border-radius: 4px; height: 10px; width: 180px; overflow: hidden; }
.bar-fill { background: #8a3b12; height: 10px; }
.footnote { color: #57606a; font-size: 12px; margin-top: 28px; line-height: 1.6; }
"""

OWNER_LABEL = {
    FailureClass.HIT: ("nobody", "owner-none"),
    FailureClass.FALSE_HIT: ("whoever chose the chunk size", "owner-chunking"),
    FailureClass.SEVERED: ("whoever chose the chunk size", "owner-chunking"),
    FailureClass.RANK_MISS: ("whoever owns the reranker", "owner-retrieval"),
    FailureClass.EMBEDDING_MISS: ("whoever owns the model", "owner-retrieval"),
    FailureClass.ABSENT: ("whoever owns ingestion", "owner-none"),
}


def _cell(label: str, value: str, note: str) -> str:
    return (
        f'<div class="cell"><div class="cell-label">{escape(label)}</div>'
        f'<div class="cell-value">{escape(value)}</div>'
        f'<div class="cell-note">{escape(note)}</div></div>'
    )


def _summary_section(result: Evaluation) -> str:
    cells = [
        _cell(
            f"recall@{result.k}, as usually scored",
            f"{result.rate(result.recall_generous)}%",
            "a retrieved chunk touched the annotated answer",
        ),
        _cell(
            "questions actually answerable",
            f"{result.rate(result.answerable)}%",
            "some chunk in the whole index holds the entire answer",
        ),
        _cell(
            "scored hits that cannot be answered",
            str(result.false_hits),
            f"{result.rate(result.false_hits, result.recall_generous)}% of the scored hits",
        ),
        _cell(
            "misses no retriever can fix",
            str(result.severed_misses),
            f"{result.rate(result.severed_misses, result.misses)}% of the recorded misses",
        ),
    ]
    return (
        '<section class="section"><h2 id="summary">Summary</h2>'
        f'<div class="grid">{"".join(cells)}</div></section>'
    )


def _taxonomy_section(result: Evaluation) -> str:
    counts = dict.fromkeys(FailureClass, 0)
    for outcome in result.outcomes:
        counts[outcome.failure] += 1
    rows = []
    widest = max(counts.values()) or 1
    for cls in FailureClass:
        count = counts[cls]
        if not count and cls is not FailureClass.HIT:
            continue
        label, css = OWNER_LABEL[cls]
        width = round(180 * count / widest)
        rows.append(
            f"<tr><td><span class=\"{css}\">{escape(str(cls))}</span></td>"
            f'<td class="numeric">{count}</td>'
            f'<td><div class="bar"><div class="bar-fill" style="width:{width}px"></div></div></td>'
            f"<td>{escape(label)}</td></tr>"
        )
    return (
        '<section class="section"><h2 id="taxonomy">Who can fix what</h2>'
        '<table class="data-table"><thead><tr><th>class</th><th class="numeric">questions</th>'
        "<th></th><th>owner</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _index_section(result: Evaluation) -> str:
    rows = [
        ("chunks in the index", str(result.chunk_count)),
        ("characters stored against corpus size", f"{result.index_multiple} times"),
        ("mean annotated answer length", f"{result.mean_answer_length} characters"),
        ("candidate pool behind the top k", str(result.pool)),
        ("questions a retrieval change could fix", str(result.retrieval_fixable)),
        ("questions only re-chunking could fix", str(result.chunking_fixable)),
    ]
    body = "".join(
        f"<tr><td>{escape(label)}</td><td class=\"numeric\">{escape(value)}</td></tr>"
        for label, value in rows
    )
    return (
        '<section class="section"><h2 id="index">What the index costs</h2>'
        '<table class="data-table"><thead><tr><th>quantity</th>'
        '<th class="numeric">value</th></tr></thead>'
        f"<tbody>{body}</tbody></table></section>"
    )


def build_report(result: Evaluation) -> str:
    """Render one evaluation as a self-contained HTML page."""
    misattributed = result.false_hits + result.severed_misses
    verdict_class = "verdict-finding" if misattributed else "verdict-clear"
    verdict = (
        f"RECALL@{result.k} MISATTRIBUTES {misattributed} QUESTIONS"
        if misattributed
        else f"RECALL@{result.k} ATTRIBUTES EVERY FAILURE CORRECTLY"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>rag failure attribution report</title>"
        f"<style>{STYLESHEET}</style></head><body><div class='page'>"
        '<h1 class="masthead">Retrieval failure attribution</h1>'
        f'<p class="subtitle">chunker {escape(result.chunker)}, k of {result.k}, '
        f"{result.total} questions over {result.chunk_count} chunks</p>"
        f'<span class="{verdict_class}">{escape(verdict)}</span>'
        f"{_summary_section(result)}{_taxonomy_section(result)}{_index_section(result)}"
        '<p class="footnote">Ground truth is a character range in a document. Retrieval returns '
        "chunks. The map between them is lossy in the direction recall@k reads it, so a miss "
        "carries no information about which of four things went wrong, and a scored hit can "
        "sit on an answer no chunk in the index contains.</p>"
        "</div></body></html>"
    )
