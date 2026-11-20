"""Module 19: turns `run_regression_suite()`'s judgment call into an
actual CI quality gate -- a non-zero exit status when a release should
be blocked.

Usage: `python -m llmqe.regression.run_and_gate`
"""

import sys

from llmqe.client import OllamaClient
from llmqe.regression.regression_suite import run_regression_suite
from llmqe.runner import build_records


def main() -> int:
    client = OllamaClient()
    current_records = build_records(client=client)
    result = run_regression_suite("v1", current_records)

    if result["should_block_release"]:
        print("QUALITY GATE FAILED -- blocking release:")
        for f in result["blocking_failures"]:
            print(f"  - {f}")
        return 1

    print(f"Quality gate passed. {result['total_diffs']} non-blocking diff(s) noted for review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
