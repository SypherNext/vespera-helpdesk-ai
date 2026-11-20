"""Module 3: the framework's first real LLM test, corrected -- asserts
on properties of a generative field instead of an exact string."""

import json

from llmqe.client import SYSTEM_PROMPT, OllamaClient

client = OllamaClient()


def test_sso_question_returns_valid_shape():
    result = client.ask_with_system(SYSTEM_PROMPT, "Does the Pro plan support SSO?")
    parsed = json.loads(result.text)  # fails the test if this isn't valid JSON

    assert set(parsed.keys()) == {"answer", "topic"}
    assert parsed["topic"] in {"billing", "security", "product", "other"}
    assert "sso" in parsed["answer"].lower() or "single sign-on" in parsed["answer"].lower()
    assert parsed["answer"].lower().startswith("yes") or "yes," in parsed["answer"].lower()
