"""The closed form, checked twice: once against its own assumption, once against the corpus.

The first check places spans uniformly at random in a synthetic document, which is exactly what
the derivation assumes. If the model is right, the residual there should be inside sampling
noise, and it is.

The second check runs the same prediction against the shipped corpus, where spans sit at
sentence boundaries inside paragraphs of similar length. The residual is small for the
non-overlapping chunkers and much larger for the overlapping one. That gap is a finding rather
than an error: it says the corpus places answers on a grid that correlates with the chunk grid,
so an overlap tuned by the formula alone will do better than the formula predicts. Publishing
the residual is the difference between a model and a slogan.
"""

from __future__ import annotations

import random

from rfa.chunking import CHUNKER_GEOMETRY, fixed_size, get_chunker, predicted_severance_rate
from rfa.corpus import AnswerSpan, Document, load_corpus
from rfa.evaluate import evaluate

from _common import CORPUS, print_table, write_result

TRIALS = 4000
SYNTHETIC_LENGTH = 20000
SPAN_LENGTHS = (150, 210, 300)


def uniform_check() -> list[dict[str, float]]:
    """Measure severance for uniformly placed spans, which is the model's own assumption."""
    rng = random.Random(20260930)
    document = Document(doc_id="synthetic", text="x" * SYNTHETIC_LENGTH)
    records: list[dict[str, float]] = []
    for size, stride in sorted(set(CHUNKER_GEOMETRY.values())):
        chunks = fixed_size(size, stride)([document])
        for length in SPAN_LENGTHS:
            severed = 0
            for _ in range(TRIALS):
                start = rng.randrange(0, document.length - length)
                span = AnswerSpan("synthetic", start, start + length)
                if not any(span.contained_by(c.start, c.end) for c in chunks):
                    severed += 1
            measured = round(100.0 * severed / TRIALS, 1)
            predicted = round(100.0 * predicted_severance_rate(length, size, stride), 1)
            records.append(
                {
                    "size": size,
                    "stride": stride,
                    "span_length": length,
                    "measured_pct": measured,
                    "predicted_pct": predicted,
                    "residual_pct": round(measured - predicted, 1),
                }
            )
    return records


def corpus_check() -> list[dict[str, float]]:
    """Run the same prediction against the real corpus, where placement is not uniform."""
    corpus = load_corpus(CORPUS)
    lengths = [question.answer.length for question in corpus.questions]
    records: list[dict[str, float]] = []
    for name, (size, stride) in sorted(CHUNKER_GEOMETRY.items()):
        result = evaluate(corpus, get_chunker(name), name, k=5)
        measured = round(100.0 * (1.0 - result.answerable / result.total), 1)
        predicted = round(
            100.0
            * sum(predicted_severance_rate(length, size, stride) for length in lengths)
            / len(lengths),
            1,
        )
        records.append(
            {
                "chunker": name,
                "measured_pct": measured,
                "predicted_pct": predicted,
                "residual_pct": round(measured - predicted, 1),
            }
        )
    return records


def main() -> None:
    uniform = uniform_check()
    corpus = corpus_check()

    print_table(
        ["size", "stride", "span", "measured", "predicted", "residual"],
        [
            [r["size"], r["stride"], r["span_length"], f"{r['measured_pct']}%",
             f"{r['predicted_pct']}%", f"{r['residual_pct']}"]
            for r in uniform
        ],
        [8, 9, 8, 12, 12, 10],
    )
    worst_uniform = max(abs(r["residual_pct"]) for r in uniform)
    print(
        f"\nlargest residual under uniform placement: {worst_uniform} points "
        f"over {TRIALS} trials"
    )
    print()

    print_table(
        ["chunker", "measured", "predicted", "residual"],
        [
            [
                r["chunker"],
                f"{r['measured_pct']}%",
                f"{r['predicted_pct']}%",
                f"{r['residual_pct']}",
            ]
            for r in corpus
        ],
        [24, 12, 12, 10],
    )
    worst_corpus = max(abs(r["residual_pct"]) for r in corpus)
    print(f"\nlargest residual on the shipped corpus: {worst_corpus} points")
    print("the model describes uniform placement exactly and the corpus only approximately")

    write_result(
        "exp04_severance_model",
        {
            "trials": TRIALS,
            "uniform": uniform,
            "corpus": corpus,
            "worst_uniform_residual_pct": worst_uniform,
            "worst_corpus_residual_pct": worst_corpus,
        },
    )


if __name__ == "__main__":
    main()
