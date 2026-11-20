"""Simple evaluation techniques (Module 6): exact matching, rule-based
validation, required-field validation, and JSON/schema validation --
generalized into one reusable `validate_response()` layer.
"""

import json
from typing import Any

REQUIRED_FIELDS = {"answer", "topic"}
ALLOWED_TOPICS = {"billing", "security", "product", "other"}


def has_required_fields(parsed: dict) -> bool:
    return set(parsed.keys()) == REQUIRED_FIELDS


def matches_schema(parsed: dict) -> list[str]:
    """Returns a list of schema violations -- empty list means valid."""
    errors = []
    if not isinstance(parsed.get("answer"), str) or not parsed["answer"].strip():
        errors.append("'answer' must be a non-empty string")
    if parsed.get("topic") not in ALLOWED_TOPICS:
        errors.append(f"'topic' must be one of {ALLOWED_TOPICS}, got {parsed.get('topic')!r}")
    return errors


def answer_contains_known_wrong_value(answer: str, wrong_values: list[str]) -> bool:
    """Rule-based hallucination check (Module 5): flags an answer that
    repeats a specific value already confirmed to be false. Deliberately
    narrow -- it only catches fabrications you already know to look for."""
    lowered = answer.lower()
    return any(wrong.lower() in lowered for wrong in wrong_values)


def validate_response(raw_text: str, known_wrong_values: list[str] | None = None) -> list[str]:
    """Runs every simple check available and returns all violations found.
    An empty list means the response passed every check this module knows
    how to run -- not a guarantee the content is actually correct."""
    errors: list[str] = []
    try:
        parsed: Any = json.loads(raw_text)
    except json.JSONDecodeError:
        return ["response is not valid JSON"]

    if not has_required_fields(parsed):
        errors.append(f"missing or extra fields: {set(parsed.keys())}")

    errors.extend(matches_schema(parsed))

    if known_wrong_values and answer_contains_known_wrong_value(parsed.get("answer", ""), known_wrong_values):
        errors.append("answer contains a known-wrong value")

    return errors
