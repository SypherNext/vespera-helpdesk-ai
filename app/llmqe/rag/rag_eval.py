"""Module 10/11: RAG-specific evaluation -- retrieval relevance,
citation honesty/correctness, and groundedness scoring."""

import json
import re
from pathlib import Path

from llmqe.rag.retriever import Retriever

DATASET_PATH = Path(__file__).parent / "datasets" / "retrieval_dataset.jsonl"


def load_retrieval_cases(path: str | Path = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_retrieval_relevance(path: str | Path = DATASET_PATH, retriever: Retriever | None = None) -> list[dict]:
    """Returns a list of relevance failures. Empty list means every
    case's expected document was retrieved in the top-k results."""
    retriever = retriever or Retriever()
    cases = load_retrieval_cases(path)
    failures = []

    for case in cases:
        results = retriever.retrieve(case["question"], top_k=2)
        retrieved_ids = [r["document"]["id"] for r in results]
        if case["expected_doc_id"] not in retrieved_ids:
            failures.append({"question": case["question"], "expected": case["expected_doc_id"], "got": retrieved_ids})

    return failures


def citation_is_honest(source_doc_id: str | None, retrieved_ids: list[str]) -> bool:
    """A citation is honest if it names a document that was actually
    retrieved -- even if it's the wrong one. Citing a document that was
    never shown to the model is a generation failure: a fabricated source."""
    return source_doc_id is None or source_doc_id in retrieved_ids


def citation_is_correct(source_doc_id: str | None, expected_doc_id: str) -> bool:
    """A citation is correct if it names the right document. This can be
    honest but wrong (cited a real retrieved doc, just not the best one)."""
    return source_doc_id == expected_doc_id


def groundedness_score(answer: str, retrieved_context: str) -> float:
    """Returns the fraction of numeric claims in the answer that are
    also present in the retrieved context. 1.0 = every number is
    traceable to the source; lower scores flag likely fabrication.

    Only catches numeric fabrication -- a fabricated non-numeric claim
    needs a judge-based groundedness check this module doesn't build,
    to stay explainable and fast enough for CI use."""
    answer_numbers = set(re.findall(r"\d+", answer))
    if not answer_numbers:
        return 1.0  # nothing numeric to check -- not proof of correctness, just nothing to flag
    context_numbers = set(re.findall(r"\d+", retrieved_context))
    grounded = answer_numbers & context_numbers
    return len(grounded) / len(answer_numbers)
