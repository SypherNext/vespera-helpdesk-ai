"""Module 14: routes a baseline diff to a decision -- block unconditionally
on a blocking category (safety, above all), judge everything else against
a tolerance threshold.
"""

from llmqe.baseline.baseline import BaselineRecord, compare_to_baseline
from llmqe.config import CONFIG


def run_regression_suite(
    baseline_name: str,
    current_records: list[BaselineRecord],
    max_new_failures: int = 0,
    block_categories: frozenset = CONFIG.block_categories,
) -> dict:
    diffs = compare_to_baseline(baseline_name, current_records)
    new_failures = [d for d in diffs if "True -> False" in d.get("change", "")]

    blocking = [d for d in new_failures if d.get("current_reason") in block_categories]
    over_threshold = len(new_failures) > max_new_failures

    return {
        "total_diffs": len(diffs),
        "diffs": diffs,
        "new_failures": new_failures,
        "blocking_failures": blocking,
        "should_block_release": bool(blocking) or over_threshold,
    }
