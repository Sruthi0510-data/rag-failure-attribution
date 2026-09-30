"""Render the committed SVGs from docs/metrics.json, so a picture cannot outlive its numbers.

Mermaid is not used, for the four reasons ADR-005 records: GitHub's lazily loaded renderer can
fail on a valid fence, HTML labels produce XML that fails silently as an <img src>, the emitted
root carries a percentage width with no height so a browser has no aspect ratio, and a
transparent background renders muted labels dark grey on near black under a dark theme.

The line chart follows the visualisation rules the repository was built against: one axis
rather than two, categorical hues assigned in fixed order and never cycled, a legend plus
direct labels so identity is never carried by colour alone, thin marks, and a recessive grid.
The palette is the validated default set, and its aqua sits below 3:1 against the surface,
which obliges the visible labels this chart already carries.

The x axis is categorical. The k values run 1, 3, 5, 10, 20, 50, 100, 200 and plotting them on
a linear axis would compress the interesting half of the chart into the first inch while
implying a spacing the data does not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "metrics.json"
SWEEP = ROOT / "docs" / "experiments" / "exp02_k_sweep.json"
OUT = ROOT / "docs" / "diagrams"

SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def _text(x, y, content, size=12, fill=INK, weight="normal", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" '
        f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(content))}</text>'
    )


def _document(width, height, body):
    """Intrinsic dimensions and a painted background, both required for an <img src> embed."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PLANE}"/>'
        f"{body}</svg>"
    )


def severance(metrics: dict) -> str:
    """A schematic of one answer meeting one chunk boundary. No chart, because there is no data."""
    parts: list[str] = []
    parts.append(_text(24, 34, "How an answer stops being an answer", 17, INK, "bold"))
    parts.append(
        _text(
            24, 56,
            f"answers average {metrics['mean_answer_length']} characters long; "
            f"chunks are 320",
            12, SECOND,
        )
    )

    left, top, width, height = 24, 86, 616, 46
    parts.append(
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="4" fill="{SURFACE}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    for index in range(1, 4):
        x = left + index * (width / 4)
        parts.append(
            f'<line x1="{x:.0f}" y1="{top - 8}" x2="{x:.0f}" y2="{top + height + 8}" '
            f'stroke="{INK}" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )
    parts.append(_text(left + 6, top - 14, "one document, cut every 320 characters", 11, MUTED))

    span_start = left + 1.35 * (width / 4)
    span_width = 1.35 * (width / 4)
    parts.append(
        f'<rect x="{span_start:.0f}" y="{top + 12}" width="{span_width:.0f}" height="22" rx="3" '
        f'fill="{ORANGE}" fill-opacity="0.22" stroke="{ORANGE}" stroke-width="1.5"/>'
    )
    parts.append(_text(span_start + 8, top + 27, "the annotated answer", 11, INK, "bold"))
    parts.append(
        _text(24, top + height + 34, "the boundary falls inside the answer, so no chunk holds "
              "all of it", 12, INK, "bold")
    )

    rows = [
        ("chunk 2", "holds the first half", "overlaps the answer"),
        ("chunk 3", "holds the second half", "overlaps the answer"),
        ("either chunk", "answers the question", "no"),
    ]
    y0 = top + height + 58
    for index, (name, holds, verdict) in enumerate(rows):
        y = y0 + index * 30
        parts.append(
            f'<rect x="24" y="{y}" width="616" height="24" rx="3" fill="{SURFACE}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(_text(38, y + 16, name, 11, INK, "bold"))
        parts.append(_text(150, y + 16, holds, 11, SECOND))
        colour = ORANGE if verdict == "no" else MUTED
        weight = "bold" if verdict == "no" else "normal"
        parts.append(_text(400, y + 16, verdict, 11, colour, weight))

    parts.append(
        _text(24, y0 + 110,
              "recall@k sees a retrieved chunk touching the annotated span and records a hit. "
              "On this corpus that", 11, SECOND)
    )
    parts.append(
        _text(24, y0 + 128,
              f"happens {metrics['false_hits']} times, which is "
              f"{metrics['false_hit_share_pct']}% of the scored hits.", 11, SECOND)
    )
    return _document(664, y0 + 152, "".join(parts))


def metric_gap(metrics: dict, points: list[dict]) -> str:
    """Three series over a categorical k axis: what the metric says against what is possible."""
    parts: list[str] = []
    width, height = 800, 420
    plot_left, plot_right = 64, 540
    plot_top, plot_bottom = 96, 318

    parts.append(_text(24, 34, "Raising k cannot raise the ceiling", 17, INK, "bold"))
    parts.append(
        _text(24, 56,
              f"{metrics['corpus_questions']} annotated questions over "
              f"{metrics['corpus_chunks']} chunks at 320 characters", 12, SECOND)
    )

    parts.append(
        f'<rect x="{plot_left - 40}" y="{plot_top - 16}" width="{plot_right - plot_left + 250}" '
        f'height="{plot_bottom - plot_top + 62}" rx="6" fill="{SURFACE}" stroke="{GRID}" '
        f'stroke-width="1"/>'
    )

    for value in (0, 25, 50, 75, 100):
        y = plot_bottom - (value / 100) * (plot_bottom - plot_top)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(_text(plot_left - 10, y + 4, f"{value}%", 10, MUTED, anchor="end"))

    count = len(points)
    step = (plot_right - plot_left) / (count - 1)

    def path_for(key: str) -> str:
        coordinates = []
        for index, point in enumerate(points):
            x = plot_left + index * step
            y = plot_bottom - (point[key] / 100) * (plot_bottom - plot_top)
            coordinates.append(f"{x:.1f},{y:.1f}")
        return " ".join(coordinates)

    # The last two series converge, so their end labels are pushed apart by hand. Overlapping
    # direct labels are worse than none: they are the thing a reader trusts least.
    series = [
        ("recall_generous_pct", ORANGE, "recall@k as scored", 0),
        ("recall_strict_pct", BLUE, "strict recall", 11),
        ("answerable_pct", AQUA, "answerable ceiling", -11),
    ]
    for key, colour, label, offset in series:
        parts.append(
            f'<polyline points="{path_for(key)}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for index, point in enumerate(points):
            x = plot_left + index * step
            y = plot_bottom - (point[key] / 100) * (plot_bottom - plot_top)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}" stroke="{SURFACE}" '
                f'stroke-width="2"/>'
            )
        last = points[-1]
        y = plot_bottom - (last[key] / 100) * (plot_bottom - plot_top) + offset
        parts.append(_text(plot_right + 12, y + 4, f"{label}  {last[key]}%", 11, colour, "bold"))

    for index, point in enumerate(points):
        x = plot_left + index * step
        parts.append(_text(x, plot_bottom + 20, point["k"], 10, MUTED, anchor="middle"))
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" '
        f'stroke="{AXIS}" stroke-width="1.5"/>'
    )
    parts.append(
        _text((plot_left + plot_right) / 2, plot_bottom + 40, "k, on a categorical axis", 11,
              MUTED, anchor="middle")
    )

    legend_y = 74
    for index, (_, colour, label, _offset) in enumerate(series):
        x = 24 + index * 190
        parts.append(f'<rect x="{x}" y="{legend_y - 8}" width="11" height="3" fill="{colour}"/>')
        parts.append(_text(x + 17, legend_y - 2, label, 11, SECOND))

    parts.append(
        _text(24, plot_bottom + 74,
              f"at k of {metrics['perfect_recall_at_k']} the metric reports a perfect score. "
              f"The ceiling has not moved:", 11, INK)
    )
    parts.append(
        _text(24, plot_bottom + 92,
              f"only {metrics['answerable_pct']}% of questions have their whole answer inside "
              f"any one chunk.", 11, INK)
    )
    return _document(width, height, "".join(parts))


def main() -> int:
    metrics = json.loads(METRICS.read_text())["metrics"]
    points = json.loads(SWEEP.read_text())["points"]
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for name, svg in (
        ("severance", severance(metrics)),
        ("metric-gap", metric_gap(metrics, points)),
    ):
        path = OUT / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "diagrams": [p.name for p in written],
                "source": "tools/render_diagrams.py",
                "palette": {"categorical": [ORANGE, BLUE, AQUA], "surface": SURFACE},
                "note": (
                    "figures are read from docs/metrics.json and the k sweep JSON, so a diagram "
                    "cannot drift from the prose"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
