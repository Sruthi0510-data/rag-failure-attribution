"""Every chunker on the same corpus, priced in the three currencies that actually trade off.

A chunker is usually chosen by one number, and there are three. Severance costs answers.
Overlap costs index size, which is storage and embedding spend. And overlap costs ranking too,
which is the one nobody expects: near-duplicate chunks compete with each other for the top k,
so the same question that a non-overlapping index answered at rank two can fall past k in an
overlapping one. This table is what a chunk size decision should be made against.
"""

from __future__ import annotations

from rfa.chunking import CHUNKERS, get_chunker
from rfa.corpus import load_corpus
from rfa.evaluate import evaluate
from rfa.failures import FailureClass

from _common import CORPUS, print_table, write_result

K = 5


def main() -> None:
    corpus = load_corpus(CORPUS)
    rows, payload = [], {}
    for name in sorted(CHUNKERS):
        result = evaluate(corpus, get_chunker(name), name, k=K)
        counts = dict.fromkeys(FailureClass, 0)
        for outcome in result.outcomes:
            counts[outcome.failure] += 1
        payload[name] = {
            "chunks": result.chunk_count,
            "index_multiple": result.index_multiple,
            "answerable_pct": result.rate(result.answerable),
            "recall_generous_pct": result.rate(result.recall_generous),
            "recall_strict_pct": result.rate(result.recall_strict),
            "false_hits": result.false_hits,
            "severed": result.severed_misses,
            "rank_misses": counts[FailureClass.RANK_MISS],
            "embedding_misses": counts[FailureClass.EMBEDDING_MISS],
        }
        rows.append(
            [
                name,
                result.chunk_count,
                f"{result.index_multiple}x",
                f"{result.rate(result.answerable)}%",
                f"{result.rate(result.recall_strict)}%",
                counts[FailureClass.RANK_MISS] + counts[FailureClass.EMBEDDING_MISS],
            ]
        )
    print_table(
        ["chunker", "chunks", "index", "answerable", "strict recall", "rank+embed misses"],
        rows,
        [24, 9, 8, 13, 16, 18],
    )
    print()
    best = max(payload, key=lambda name: payload[name]["recall_strict_pct"])
    print(f"best strict recall on this corpus: {best} at {payload[best]['recall_strict_pct']}%")
    write_result("exp03_chunker_tradeoff", {"k": K, "chunkers": payload, "best_strict": best})


if __name__ == "__main__":
    main()
