"""Capture this repository's own commands and commit the transcript.

The command list is the README's claims, in the order the README makes them. If
a claim changes, the list changes with it, and the capture fails the build when
a command does not exit the way the README says it does. The four documented
exit codes are therefore demonstrated rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_terminal import Step, capture  # noqa: E402

PY = sys.executable

STEPS = [
    Step(
        label="the suite, with branch coverage",
        argv=[PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--cov=rfa",
              "--cov-branch", "--cov-report=term-missing", "--cov-fail-under=100"],
        max_lines=16,
        tail_lines=4,
    ),
    Step(
        label="classify every question under the shipped chunker: a finding",
        argv=[PY, "-m", "rfa.cli", "evaluate"],
        expect=3,
        max_lines=16,
        tail_lines=5,
    ),
    Step(
        label="one question the metric calls a hit and the answer cannot be read from",
        argv=[PY, "-m", "rfa.cli", "explain", "q-0007"],
        expect=3,
        max_lines=10,
    ),
    Step(
        label="every chunker side by side",
        argv=[PY, "-m", "rfa.cli", "compare"],
        expect=3,
        max_lines=13,
    ),
    Step(
        label="an evaluation set smaller than the caller required: a refusal",
        argv=[PY, "-m", "rfa.cli", "evaluate", "--require-questions", "99999"],
        expect=4,
        max_lines=5,
        tail_lines=2,
    ),
    Step(
        label="a chunker that is not one of the seven",
        argv=[PY, "-m", "rfa.cli", "evaluate", "--chunker", "sentence"],
        expect=2,
        max_lines=4,
    ),
    Step(
        label="every published figure re-checked against the prose",
        argv=[PY, "tools/check_numbers.py"],
        max_lines=3,
    ),
]


def main() -> int:
    return capture(
        STEPS,
        title="rag-failure-attribution: captured by tools/proof_of_work.py, exit codes as returned",
        out_svg=ROOT / "docs" / "img" / "proof-of-work.svg",
        out_json=ROOT / "docs" / "proof-of-work.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
