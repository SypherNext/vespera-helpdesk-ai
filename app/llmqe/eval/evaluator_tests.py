"""Module 8: the evaluator needs its own test suite, separate from the
tests that check the assistant itself. Runs `score_case()` against a
human-labeled known-good/known-bad dataset and fails loudly if the
evaluator disagrees with human judgment.
"""

import json
from pathlib import Path

from llmqe.eval.scorer import score_case

DATASET_PATH = Path(__file__).parent / "datasets" / "evaluator_calibration.jsonl"


def load_calibration_cases(path: str | Path = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_evaluator_calibration(path: str | Path = DATASET_PATH) -> list[dict]:
    """Returns a list of mismatches between the evaluator's verdict and
    the human-assigned expected_verdict. An empty list means the
    evaluator agreed with every labeled case."""
    cases = load_calibration_cases(path)
    mismatches = []

    for case in cases:
        raw_response = json.dumps({"answer": case["candidate_answer"], "topic": "product"})
        result = score_case(case["question"], case["reference_answer"], raw_response)
        actual = "pass" if result["passed"] else "fail"

        if actual != case["expected_verdict"]:
            mismatches.append(
                {
                    "label": case["label"],
                    "expected": case["expected_verdict"],
                    "actual": actual,
                    "detail": result,
                }
            )

    return mismatches
