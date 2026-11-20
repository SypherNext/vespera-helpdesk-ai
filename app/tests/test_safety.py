"""Module 12: the adversarial safety-test suite."""

from llmqe.safety.safety_tests import test_resists_injected_instructions as _resists  # noqa: F401


def test_resists_injected_instructions():
    _resists()
