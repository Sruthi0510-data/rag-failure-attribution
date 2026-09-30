"""A seeded corpus of policy documents with answers annotated where a human would put them.

Three properties this generator has to deliver, and one it must not.

It must place answer spans without regard to any chunk boundary, because that is what a human
annotator does: they read a document and highlight the sentence, months before anyone picks a
chunk size. The generator therefore builds the document text first and records the offsets the
fact sentence happened to land on.

It must produce answers of varying length, because severance scales with the ratio of answer
length to chunk size and a corpus of uniform answers would make the relationship invisible.

It must produce questions that a lexical retriever genuinely fails on sometimes, or the rank
classes would be empty and the taxonomy would be asserted rather than exercised. Each question
is a paraphrase, and a controlled share of its content words are replaced with synonyms that do
not appear in the document at all.

What it must not do is arrange for severance. The spans fall where the prose puts them, the
chunkers cut where their arithmetic says, and the severance rate that comes out is compared
against the closed form in chunking.py rather than dialled in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rfa.corpus import AnswerSpan, Corpus, Document, Question
from rfa.errors import ConfigError

SYSTEMS = ("Ledger", "Atlas", "Beacon", "Corvus", "Delta Vault", "Ember", "Foundry")
REGIONS = ("EMEA", "APAC", "LATAM", "North America", "the Nordics", "the United Kingdom")
ARTEFACTS = ("audit logs", "customer records", "transaction receipts", "access reviews",
             "incident reports", "consent records", "settlement files")
TEAMS = ("Platform Engineering", "Risk", "Data Governance", "Security Operations", "Finance")

FACT_TEMPLATES = (
    "The retention period for {artefact} held in {system} for {region} is {days} days, "
    "after which the records are purged automatically by the nightly job.",
    "Access to {artefact} in {system} for {region} is reviewed by {team} every {days} days, "
    "and any account that fails two consecutive reviews is disabled.",
    "{team} must approve any export of {artefact} from {system} covering {region} before it "
    "leaves the production boundary, and the approval expires after {days} days.",
    "{system} replicates {artefact} for {region} to the secondary site within {days} minutes, "
    "and a lag beyond that threshold pages the on-call engineer.",
)

CONTINUATIONS = (
    "The same limit applies to any derived extract, including the reporting copy held by "
    "the analytics team.",
    "Where the window is exceeded, the owning team raises a finding and records the breach "
    "in the control register within one working day.",
    "This figure is fixed in the regional addendum and cannot be varied locally without a "
    "governance exception.",
)
"""Second sentences that some facts carry.

They exist because the first version of this corpus made every answer exactly one sentence
long, and a sentence-aligned chunker scored a perfect hundred on it by construction. That is
not a finding about sentence chunking, it is a finding about a corpus whose annotation unit
happened to match the chunker's unit. A share of answers now spans two sentences, which is what
a real annotation looks like when the answer is a rule plus its exception, and the sentence
chunker has to earn its result.
"""

QUESTION_TEMPLATES = (
    "How long are {artefact} kept in {system} for {region}?",
    "Who signs off on {artefact} leaving {system} in {region}?",
    "What is the review cadence for {artefact} access in {system} for {region}?",
    "What replication delay is tolerated for {artefact} in {system} for {region}?",
)

SYNONYMS = {
    "retention": "storage lifetime",
    "purged": "expunged",
    "reviewed": "reassessed",
    "approve": "authorise",
    "export": "extraction",
    "replicates": "mirrors",
    "records": "entries",
    "audit logs": "activity trails",
    "customer records": "client entries",
    "transaction receipts": "payment confirmations",
    "access reviews": "entitlement checks",
    "incident reports": "outage write-ups",
    "consent records": "permission entries",
    "settlement files": "clearing batches",
}

FILLER = (
    "This section is maintained by the owning team and is reviewed at each quarterly checkpoint.",
    "Exceptions require a documented rationale and a named accountable owner.",
    "Changes take effect at the start of the following calendar month unless stated otherwise.",
    "Where a local regulation is stricter, the local regulation applies and supersedes this text.",
    "Metrics for this control are published to the governance dashboard on the first working day.",
    "Historic versions of this page remain available in the document store for seven years.",
    "Questions about this policy should be raised in the governance channel rather than by email.",
    "The control owner confirms applicability each year and records it in the control register.",
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Shape of the corpus. Every field is documented in the README fixture table."""

    documents: int = 120
    questions: int = 360
    seed: int = 20260930
    paragraphs_per_document: int = 5
    sentences_per_paragraph: int = 4
    paraphrase_rate: float = 0.8
    multi_sentence_rate: float = 0.45

    def __post_init__(self) -> None:
        if self.documents < 1:
            raise ConfigError("documents must be at least 1")
        if self.questions < 1:
            raise ConfigError("questions must be at least 1")
        if self.paragraphs_per_document < 1:
            raise ConfigError("paragraphs_per_document must be at least 1")
        if self.sentences_per_paragraph < 2:
            raise ConfigError("sentences_per_paragraph must be at least 2 to hold a fact")
        for name in ("paraphrase_rate", "multi_sentence_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must be between 0 and 1")


def _paraphrase(text: str, rng: random.Random, rate: float) -> str:
    """Replace content words with synonyms absent from the corpus, at the configured rate."""
    for original, replacement in SYNONYMS.items():
        if original in text and rng.random() < rate:
            text = text.replace(original, replacement)
    return text


def generate(config: GeneratorConfig | None = None) -> Corpus:
    """Build a deterministic corpus and question set."""
    config = config or GeneratorConfig()
    rng = random.Random(config.seed)

    documents: list[Document] = []
    planted: list[tuple[str, int, int, dict[str, str], int]] = []

    for index in range(config.documents):
        doc_id = f"policy-{index:04d}"
        fact_paragraph = rng.randrange(config.paragraphs_per_document)
        fact_position = rng.randrange(config.sentences_per_paragraph)
        fields = {
            "system": rng.choice(SYSTEMS),
            "region": rng.choice(REGIONS),
            "artefact": rng.choice(ARTEFACTS),
            "team": rng.choice(TEAMS),
            "days": str(rng.choice((7, 14, 30, 45, 90, 180, 365))),
        }
        template_index = rng.randrange(len(FACT_TEMPLATES))
        fact = FACT_TEMPLATES[template_index].format(**fields)
        if rng.random() < config.multi_sentence_rate:
            fact = fact + " " + rng.choice(CONTINUATIONS)

        parts: list[str] = []
        cursor = 0
        fact_start = fact_end = -1
        for paragraph_index in range(config.paragraphs_per_document):
            sentences: list[str] = []
            for sentence_index in range(config.sentences_per_paragraph):
                if paragraph_index == fact_paragraph and sentence_index == fact_position:
                    sentences.append(fact)
                else:
                    sentences.append(rng.choice(FILLER))
            block = " ".join(sentences)
            if paragraph_index == fact_paragraph:
                offset = block.index(fact)
                fact_start = cursor + offset
                fact_end = fact_start + len(fact)
            parts.append(block)
            cursor += len(block) + 2

        text = "\n\n".join(parts)
        documents.append(
            Document(doc_id=doc_id, text=text, title=f"{fields['system']} policy {index:04d}")
        )
        planted.append((doc_id, fact_start, fact_end, fields, template_index))

    questions: list[Question] = []
    for number in range(config.questions):
        doc_id, start, end, fields, template_index = planted[number % len(planted)]
        template = QUESTION_TEMPLATES[(template_index + number // len(planted)) % len(
            QUESTION_TEMPLATES
        )]
        text = _paraphrase(template.format(**fields), rng, config.paraphrase_rate)
        questions.append(
            Question(
                question_id=f"q-{number:04d}",
                text=text,
                answer=AnswerSpan(doc_id=doc_id, start=start, end=end),
            )
        )

    return Corpus(documents=tuple(documents), questions=tuple(questions))
