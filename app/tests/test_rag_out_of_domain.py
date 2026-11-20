"""Module 10/11: retrieval relevance, and generation behavior when
retrieval comes up with nothing genuinely relevant."""

from llmqe.rag.rag_eval import test_retrieval_relevance as _retrieval_relevance_check


def test_retrieval_relevance():
    failures = _retrieval_relevance_check()
    assert not failures, f"Retrieval relevance failures: {failures}"
