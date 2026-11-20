"""Module 16: Functional Quality vs. Operational Quality -- latency,
timeouts, retries, and concurrent-request behavior. Carries the fix for
the real bug this module's lesson diagnoses: a caller that builds a new
Retriever() (which re-embeds the whole knowledge base) on every single
request instead of reusing one already-built instance.
"""

import time
from enum import Enum

from llmqe.client import OllamaClient
from llmqe.rag.assistant import RAG_SYSTEM_PROMPT, ask_with_retrieval
from llmqe.rag.retriever import Retriever


def measure_latency(fn, *args, **kwargs) -> tuple[float, object]:
    start = time.time()
    result = fn(*args, **kwargs)
    return time.time() - start, result


class FailureCategory(Enum):
    TIMEOUT = "timeout"
    ERROR = "error"
    SLOW = "slow"  # succeeded, but past the latency budget
    WRONG_ANSWER = "wrong_answer"  # an evaluation failure, not this module's concern


def classify_call(fn, *args, timeout_seconds: float = 5.0, latency_budget: float = 3.0, **kwargs) -> dict:
    start = time.time()
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        return {"category": FailureCategory.ERROR, "detail": str(e)}

    latency = time.time() - start
    if latency > timeout_seconds:
        return {"category": FailureCategory.TIMEOUT, "latency": latency}
    if latency > latency_budget:
        return {"category": FailureCategory.SLOW, "latency": latency}
    return {"category": None, "latency": latency, "result": result}


def call_with_retry(fn, *args, max_attempts: int = 3, **kwargs):
    """Only ever appropriate for FailureCategory.TIMEOUT/ERROR -- never
    for a WRONG_ANSWER, which a retry can't fix and would only hide."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            time.sleep(2**attempt)  # exponential backoff
    raise last_error


def ask_plain(client: OllamaClient | None = None, question: str = "Does the Pro plan support SSO?"):
    client = client or OllamaClient()
    return client.ask_with_system(RAG_SYSTEM_PROMPT, question)


def ask_rag_reusing_retriever(
    retriever: Retriever,
    client: OllamaClient | None = None,
    question: str = "Does the Pro plan support SSO?",
):
    """The fixed shape: retriever built once, outside the per-request
    path, and reused. Never construct Retriever() inside this function."""
    client = client or OllamaClient()
    return ask_with_retrieval(question, retriever, client)
