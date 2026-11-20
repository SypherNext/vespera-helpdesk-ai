"""Module 5: the first Vespera incident -- a rule-based hallucination
check that documents a known, real failure until Module 9's retrieval
fix actually addresses the root cause (tests/test_rag_fundamentals.py
confirms the fix)."""

import json

from llmqe.client import SYSTEM_PROMPT, OllamaClient
from llmqe.validation import answer_contains_known_wrong_value


def test_free_plan_rate_limit_is_not_fabricated_once_grounded():
    """Without retrieval, the plain assistant is expected to sometimes
    fabricate a rate limit -- that's Module 5's whole point. This test
    exercises the known-wrong-value check itself, not a live call,
    so it is deterministic regardless of what the ungrounded model
    happens to say on a given run."""
    known_wrong_values = ["500 requests", "1000 requests", "unlimited"]
    fabricated = json.dumps({"answer": "The Free plan allows 500 requests per minute.", "topic": "product"})
    parsed = json.loads(fabricated)
    assert answer_contains_known_wrong_value(parsed["answer"], known_wrong_values)


def test_live_call_against_known_facts():
    """A live, non-asserting probe you can run by hand to see today's
    ungrounded answer -- kept informational rather than a hard assertion,
    since an un-grounded plain LLM call is *expected* to vary."""
    client = OllamaClient()
    result = client.ask_with_system(SYSTEM_PROMPT, "What's the API rate limit on the Free plan?")
    parsed = json.loads(result.text)
    assert isinstance(parsed["answer"], str) and parsed["answer"].strip()
