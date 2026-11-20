"""Module 13: run the evaluation suite once against a known-good state
and record it as the `v1` baseline. Run this once, deliberately, not on
every test run -- a baseline is a recorded snapshot to compare against,
not something that regenerates itself silently.

Usage: `python scripts/build_baseline.py`
"""

from llmqe.baseline.baseline import record_baseline
from llmqe.client import OllamaClient
from llmqe.runner import build_records


def main() -> None:
    client = OllamaClient()
    records = build_records(client=client)
    path = record_baseline("v1", records)
    print(f"Baseline written to {path}")
    for r in records:
        print(f"  {r.test_case!r}: passed={r.passed} stage={r.reason}")


if __name__ == "__main__":
    main()
