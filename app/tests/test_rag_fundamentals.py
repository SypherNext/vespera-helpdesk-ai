"""Module 9: RAG grounds the answer that Module 5 showed being fabricated."""

from llmqe.client import OllamaClient
from llmqe.rag.assistant import ask_with_retrieval
from llmqe.rag.retriever import Retriever


def test_rate_limit_question_is_grounded_by_retrieval():
    client = OllamaClient()
    retriever = Retriever()
    result = ask_with_retrieval("What's the API rate limit on the Free plan?", retriever, client)

    assert "free-plan" in [r["document"]["id"] for r in result["retrieved"]]
