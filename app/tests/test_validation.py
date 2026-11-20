"""Module 6: the reusable validate_response() layer."""

from llmqe.client import SYSTEM_PROMPT, OllamaClient
from llmqe.validation import validate_response


def test_free_plan_rate_limit_question():
    client = OllamaClient()
    result = client.ask_with_system(SYSTEM_PROMPT, "What's the API rate limit on the Free plan?")

    errors = validate_response(result.text, known_wrong_values=["500 requests", "1000 requests", "unlimited"])
    assert errors == [], f"Validation failed: {errors}"
