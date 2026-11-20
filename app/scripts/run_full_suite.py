"""Module 19's nightly full-evaluation entry point: runs the evaluator
calibration check, the RAG retrieval-relevance check, logs a full
evaluation pass, runs the regression gate against the `v1` baseline, and
writes the human-readable report. Exits non-zero if the gate blocks.

Usage: `python scripts/run_full_suite.py`
"""

import sys

from llmqe.client import OllamaClient
from llmqe.eval.evaluator_tests import run_evaluator_calibration
from llmqe.rag.rag_eval import test_retrieval_relevance
from llmqe.regression.regression_suite import run_regression_suite
from llmqe.reporting.report import write_report
from llmqe.runner import DEFAULT_CASES, run_and_log
from llmqe.safety.safety_tests import run_injection_cases


def main() -> int:
    client = OllamaClient()

    print("== Evaluator calibration ==")
    calibration_mismatches = run_evaluator_calibration()
    print(f"{len(calibration_mismatches)} mismatch(es)")

    print("== RAG retrieval relevance ==")
    relevance_failures = test_retrieval_relevance()
    print(f"{len(relevance_failures)} failure(s)")

    print("== Safety suite ==")
    safety_failures = run_injection_cases()
    print(f"{len(safety_failures)} failure(s)")

    print("== Logging evaluation runs ==")
    current_records = []
    for case in DEFAULT_CASES:
        entry = run_and_log(case["question"], case["reference_answer"], client)
        current_records.append(entry)
        print(f"  {entry.test_case_id!r}: passed={entry.passed}")

    print("== Regression gate ==")
    from llmqe.runner import build_records

    baseline_style_records = build_records(client=client)
    gate_result = run_regression_suite("v1", baseline_style_records)
    print(f"should_block_release={gate_result['should_block_release']}")

    print("== Report ==")
    report = write_report(diffs=gate_result["diffs"])
    print(report)

    if calibration_mismatches or safety_failures or gate_result["should_block_release"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
