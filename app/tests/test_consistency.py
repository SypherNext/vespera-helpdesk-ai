"""Module 4: consistency across repeated, realistic (non-pinned) calls."""

import json

import pytest

from llmqe.client import SYSTEM_PROMPT, OllamaClient


@pytest.mark.slow
def test_sso_answer_is_consistent_across_runs():
    client = OllamaClient(temperature=0.8, seed=None)  # realistic, non-pinned settings
    answers = []

    for _ in range(5):
        result = client.ask_with_system(SYSTEM_PROMPT, "Does the Pro plan support SSO?")
        parsed = json.loads(result.text)
        answers.append(parsed["answer"].lower())

    says_yes = ["yes" in a or "supports sso" in a for a in answers]
    mentions_sso = ["sso" in a or "single sign-on" in a for a in answers]

    assert all(says_yes), f"Inconsistent yes/no answers across runs: {answers}"
    assert all(mentions_sso), f"Some answers dropped the SSO detail entirely: {answers}"
