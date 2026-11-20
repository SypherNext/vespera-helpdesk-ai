"""Module 8: the evaluator needs its own test suite, separate from the
tests that check the assistant itself."""

from llmqe.eval.evaluator_tests import run_evaluator_calibration


def test_evaluator_matches_human_judgment():
    mismatches = run_evaluator_calibration()
    assert not mismatches, f"Evaluator disagreed with human judgment on {len(mismatches)} case(s): {mismatches}"
