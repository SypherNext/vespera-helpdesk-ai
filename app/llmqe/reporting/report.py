"""Module 18 + Module 23's report-clarity fix: turns eval_log.jsonl into
a short, scannable Markdown report, with new-test-case diffs (no prior
baseline, not a regression) called out separately from real failures.
"""

from collections import Counter
from pathlib import Path

from llmqe.observability.logger import read_log

DEFAULT_REPORT_PATH = Path(__file__).parent / "latest_report.md"


def summarize(entries: list[dict]) -> dict:
    total = len(entries)
    passed = sum(1 for e in entries if e["passed"])
    failures = [e for e in entries if not e["passed"]]
    by_category = Counter(e["failure_category"] for e in failures)
    avg_latency = sum(e["latency_seconds"] for e in entries) / total if total else 0

    return {
        "total_runs": total,
        "passed": passed,
        "failed": len(failures),
        "pass_rate": passed / total if total else 0,
        "failures_by_category": dict(by_category),
        "average_latency_seconds": avg_latency,
        "failures": failures,
    }


def render_report(summary: dict, diffs: list[dict] | None = None) -> str:
    lines = [
        "# LLM Quality & Evaluation Framework -- Run Report",
        "",
        f"**Total runs:** {summary['total_runs']}",
        f"**Passed:** {summary['passed']} ({summary['pass_rate']:.0%})",
        f"**Failed:** {summary['failed']}",
        f"**Average latency:** {summary['average_latency_seconds']:.2f}s",
        "",
        "## Failures by category",
        "",
    ]
    if not summary["failures_by_category"]:
        lines.append("None.")
    for category, count in summary["failures_by_category"].items():
        lines.append(f"- **{category}**: {count}")

    if summary["failures"]:
        lines += ["", "## Failure details", ""]
        for f in summary["failures"]:
            lines.append(f"- `{f['test_case_id']}` -- {f['failure_category']} -- {f['reason']}")

    diffs = diffs or []
    new_cases = [d for d in diffs if "no prior baseline" in d.get("change", "")]
    if new_cases:
        lines += ["", "## New test cases (no baseline yet -- not a regression)", ""]
        for d in new_cases:
            lines.append(f"- `{d['test_case']}`")

    return "\n".join(lines)


def write_report(path: str | Path = DEFAULT_REPORT_PATH, diffs: list[dict] | None = None) -> str:
    entries = read_log()
    summary = summarize(entries)
    report = render_report(summary, diffs)
    with open(path, "w") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    print(write_report())
