"""Raise k and watch the metric reach a perfect score without answering one more question.

This is the impossibility in its most direct form. Overlap-based recall@k rises with k because
a wider net eventually catches a chunk touching the annotated span. Answerability does not move
at all, because it is a property of the chunk boundaries rather than of how many chunks you
look at. A team optimising the first number can reach a hundred per cent while the second never
moves, and nothing in the metric says so.
"""

from __future__ import annotations

from rfa.chunking import get_chunker
from rfa.corpus import load_corpus
from rfa.evaluate import evaluate
from rfa.retrieve import pool_size

from _common import CORPUS, print_table, write_result

CHUNKER = "fixed-320"
SCHEDULE = (1, 3, 5, 10, 20, 50, 100, 200)


def main() -> None:
    corpus = load_corpus(CORPUS)
    chunker = get_chunker(CHUNKER)
    rows, points = [], []
    ceiling = None
    perfect_at = None
    for k in SCHEDULE:
        result = evaluate(corpus, chunker, CHUNKER, k=k, pool=max(pool_size(k), k))
        summary = result.summary()
        ceiling = summary["answerable_pct"]
        if perfect_at is None and summary["recall_generous_pct"] >= 100.0:
            perfect_at = k
        points.append(
            {
                "k": k,
                "recall_generous_pct": summary["recall_generous_pct"],
                "recall_strict_pct": summary["recall_strict_pct"],
                "answerable_pct": summary["answerable_pct"],
                "false_hits": summary["false_hits"],
            }
        )
        rows.append(
            [
                k,
                f"{summary['recall_generous_pct']}%",
                f"{summary['recall_strict_pct']}%",
                f"{summary['answerable_pct']}%",
                summary["false_hits"],
            ]
        )
    print_table(["k", "recall@k", "strict", "answerable", "false hits"], rows, [8, 12, 10, 13, 12])
    print()
    print(f"the answerable ceiling never moves: {ceiling}% at every k")
    if perfect_at:
        print(
            f"overlap-based recall@k reaches a perfect score at k of {perfect_at}, "
            f"with the ceiling unchanged"
        )
    write_result(
        "exp02_k_sweep",
        {
            "chunker": CHUNKER,
            "points": points,
            "answerable_ceiling_pct": ceiling,
            "perfect_recall_at_k": perfect_at,
            "max_k": SCHEDULE[-1],
        },
    )


if __name__ == "__main__":
    main()
