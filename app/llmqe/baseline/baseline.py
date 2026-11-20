"""Module 13: recording and comparing against a baseline -- a snapshot
of expected behavior, so a future run can be compared against something
concrete instead of a vague sense of 'it usually works.'
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

BASELINE_DIR = Path(__file__).parent / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BaselineRecord:
    model: str
    config: dict
    prompt_version: str
    dataset_version: str
    evaluator_version: str
    test_case: str
    passed: bool
    score: float | None
    reason: str
    latency_seconds: float
    timestamp: float
    failure_category: str | None


def record_baseline(name: str, records: list[BaselineRecord]) -> Path:
    path = BASELINE_DIR / f"{name}.json"
    path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    return path


def load_baseline(name: str) -> list[dict]:
    path = BASELINE_DIR / f"{name}.json"
    return json.loads(path.read_text())


def compare_to_baseline(baseline_name: str, current_records: list[BaselineRecord]) -> list[dict]:
    """Returns a list of differences between the baseline and a new run,
    keyed by test_case. Doesn't judge whether a difference is good or
    bad -- that's the regression suite's job."""
    baseline = {r["test_case"]: r for r in load_baseline(baseline_name)}
    diffs = []

    for record in current_records:
        prior = baseline.get(record.test_case)
        if prior is None:
            diffs.append({"test_case": record.test_case, "change": "new test case, no prior baseline"})
            continue
        if prior["passed"] != record.passed:
            diffs.append(
                {
                    "test_case": record.test_case,
                    "change": f"pass/fail changed: {prior['passed']} -> {record.passed}",
                    "prior_reason": prior["reason"],
                    "current_reason": record.reason,
                }
            )

    return diffs
