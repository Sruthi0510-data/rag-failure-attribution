"""What overlap buys, what it costs, and the thing it does that nobody advertises.

Overlap is the standard advice once severance is understood, and the advice is right. What the
advice leaves out is that overlap does not remove a failure, it moves one. Near-duplicate
chunks compete with each other for the top k, so questions that were severed become answerable
and some of them arrive as rank misses instead. That is progress, because a rank miss is
fixable by a reranker and a severed answer is fixable by nothing, but it is not the free win
the advice implies and it changes what the team should buy next.
"""

from __future__ import annotations

from rfa.chunking import get_chunker
from rfa.corpus import load_corpus
from rfa.evaluate import evaluate
from rfa.failures import FailureClass

from _common import CORPUS, print_table, write_result

PAIRS = (("fixed-320", "fixed-320-overlap-80"), ("sentence-3", "sentence-3-overlap-1"))
K = 5


def _profile(corpus, name: str) -> dict[str, float]:
    result = evaluate(corpus, get_chunker(name), name, k=K)
    counts = dict.fromkeys(FailureClass, 0)
    for outcome in result.outcomes:
        counts[outcome.failure] += 1
    return {
        "chunks": result.chunk_count,
        "index_multiple": result.index_multiple,
        "answerable_pct": result.rate(result.answerable),
        "recall_strict_pct": result.rate(result.recall_strict),
        "unanswerable": result.total - result.answerable,
        "rank_and_embedding_misses": counts[FailureClass.RANK_MISS]
        + counts[FailureClass.EMBEDDING_MISS],
    }


def main() -> None:
    corpus = load_corpus(CORPUS)
    rows, payload = [], []
    for plain_name, overlap_name in PAIRS:
        plain = _profile(corpus, plain_name)
        overlap = _profile(corpus, overlap_name)
        record = {
            "plain": plain_name,
            "overlapping": overlap_name,
            "unanswerable_removed": plain["unanswerable"] - overlap["unanswerable"],
            "rank_misses_added": overlap["rank_and_embedding_misses"]
            - plain["rank_and_embedding_misses"],
            "strict_recall_gain_pct": round(
                overlap["recall_strict_pct"] - plain["recall_strict_pct"], 1
            ),
            "index_multiple": overlap["index_multiple"],
            "plain_profile": plain,
            "overlap_profile": overlap,
        }
        payload.append(record)
        rows.append(
            [
                f"{plain_name} to {overlap_name}",
                record["unanswerable_removed"],
                record["rank_misses_added"],
                f"{record['strict_recall_gain_pct']:+}",
                f"{overlap['index_multiple']}x",
            ]
        )
    print_table(
        ["change", "answers recovered", "rank misses added", "strict recall", "index"],
        rows,
        [42, 19, 20, 15, 8],
    )
    print()
    converted = sum(record["rank_misses_added"] for record in payload)
    recovered = sum(record["unanswerable_removed"] for record in payload)
    print(
        f"across both pairs overlap recovered {recovered} unanswerable questions and turned "
        f"{converted} of them into rank misses"
    )
    print("a rank miss is fixable by a reranker; a severed answer is fixable by nothing")
    write_result(
        "exp05_overlap_cost",
        {
            "k": K,
            "pairs": payload,
            "answers_recovered": recovered,
            "rank_misses_added": converted,
        },
    )


if __name__ == "__main__":
    main()
