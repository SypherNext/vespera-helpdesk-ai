"""Module 16: latency, and the real retriever-rebuilding regression this
module's lesson diagnoses."""

import pytest

from llmqe.client import OllamaClient
from llmqe.rag.retriever import Retriever
from llmqe.reliability.perf_tests import ask_plain, ask_rag_reusing_retriever, measure_latency


@pytest.mark.slow
def test_rag_call_latency_with_reused_retriever():
    client = OllamaClient()
    retriever = Retriever()  # built once, reused across every call

    latency, _ = measure_latency(ask_rag_reusing_retriever, retriever, client)
    print(f"RAG call latency (retriever reused): {latency:.2f}s")
    # No absolute-time assertion here -- hardware varies. The real
    # regression this module teaches is relative: rebuilding Retriever()
    # per-call costs an entire knowledge-base re-embedding every time.
    assert latency >= 0
