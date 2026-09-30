"""rfa: attribute RAG retrieval failures to the thing that actually caused them.

The package exists because recall@k is computed over chunks while the ground truth is
annotated on documents. That mapping loses information in one direction, so a miss cannot say
whether the retriever failed or the chunker cut the answer in half, and those two have
different fixes.
"""

from rfa.corpus import AnswerSpan, Document, Question
from rfa.failures import FailureClass

__all__ = ["AnswerSpan", "Document", "FailureClass", "Question"]
__version__ = "0.1.0"
