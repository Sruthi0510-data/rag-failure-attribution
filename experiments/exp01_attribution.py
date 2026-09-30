"""Who owns each failure, under the chunk size a team is most likely to be running.

The question the README asks: when recall@k is low, whose problem is it? The usual answer is
the retrieval team's, because the metric has retrieval in the name. This experiment splits the
same questions by what would actually fix them, and the split does not favour retrieval.
"""

from __future__ import annotations

from rfa.chunking import get_chunker
from rfa.corpus import load_corpus
from rfa.evaluate import evaluate
from rfa.failures import CHUNKING_FIXABLE, RETRIEVAL_FIXABLE, FailureClass

from _common import CORPUS, print_table, write_result

CHUNKER = "fixed-320"
K = 5


def main() -> None:
    corpus = load_corpus(CORPUS)
    result = evaluate(corpus, get_chunker(CHUNKER), CHUNKER, k=K)
    counts = dict.fromkeys(FailureClass, 0)
    for outcome in result.outcomes:
        counts[outcome.failure] += 1

    owner = {
        FailureClass.HIT: "nobody",
        FailureClass.FALSE_HIT: "chunking",
        FailureClass.SEVERED: "chunking",
        FailureClass.RANK_MISS: "retrieval",
        FailureClass.EMBEDDING_MISS: "retrieval",
        FailureClass.ABSENT: "ingestion",
    }
    rows = [
        [str(cls), counts[cls], f"{result.rate(counts[cls])}%", owner[cls]]
        for cls in FailureClass
        if counts[cls] or cls is FailureClass.HIT
    ]
    print_table(["class", "questions", "share", "owner"], rows, [18, 12, 10, 12])
    print()
    print(f"recall@{K} as usually scored: {result.rate(result.recall_generous)}%")
    print(f"questions actually answerable: {result.rate(result.answerable)}%")
    print(
        f"of the {result.misses} recorded misses, {result.severed_misses} "
        f"({result.rate(result.severed_misses, result.misses)}%) need a chunking change"
    )
    print(
        f"of the {result.recall_generous} scored hits, {result.false_hits} "
        f"({result.rate(result.false_hits, result.recall_generous)}%) cannot be answered at all"
    )

    write_result(
        "exp01_attribution",
        {
            "chunker": CHUNKER,
            "k": K,
            "summary": result.summary(),
            "counts": {str(cls): counts[cls] for cls in FailureClass},
            "retrieval_fixable": sum(counts[cls] for cls in RETRIEVAL_FIXABLE),
            "chunking_fixable": sum(counts[cls] for cls in CHUNKING_FIXABLE),
        },
    )


if __name__ == "__main__":
    main()
