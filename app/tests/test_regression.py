"""Module 14: a prompt regression (the shortened system prompt) should
fail against the exact cases the hardened prompt was built to resist.
Keeps the safety suite honest -- Module 12's safety tests need to run as
part of every regression suite, not just once after they were written.
"""

from llmqe.safety.safety_tests import INJECTION_CASES, SAFETY_SYSTEM_PROMPT, SHORTENED_SYSTEM_PROMPT, run_injection_cases


def test_shortened_prompt_reintroduces_the_injection_failure():
    failures = run_injection_cases(system_prompt=SHORTENED_SYSTEM_PROMPT, cases=INJECTION_CASES)
    assert failures, "Expected the shortened prompt to regress on at least one injection case"


def test_hardened_prompt_does_not_regress():
    failures = run_injection_cases(system_prompt=SAFETY_SYSTEM_PROMPT, cases=INJECTION_CASES)
    assert not failures, f"Hardened prompt unexpectedly failed: {failures}"
