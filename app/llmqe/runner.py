"""Shared helpers combining the client, scorer, baseline, and logger --
used by build_baseline (Module 13), run_and_log (Module 17), and the CI
entry points (Module 19). Kept in one place instead of duplicated between
scripts, per Module 20's DRY fix.
"""

import time

from llmqe.baseline.baseline import BaselineRecord
from llmqe.client import SYSTEM_PROMPT, OllamaClient
from llmqe.config import CONFIG
from llmqe.eval.scorer import score_case
from llmqe.observability.logger import EvalLogEntry, log_eval

DEFAULT_CASES = [
    {
        "question": "Does the Pro plan support SSO?",
        "reference_answer": "Yes, the Pro plan includes SSO support for enterprise identity providers.",
    },
    {
        "question": "What's the API rate limit on the Free plan?",
        "reference_answer": "The Free plan is limited to 60 API requests per minute.",
    },
]


def run_one(question: str, reference_answer: str, client: OllamaClient) -> tuple[BaselineRecord, EvalLogEntry]:
    start = time.time()
    result = client.ask_with_system(SYSTEM_PROMPT, question)
    latency = time.time() - start

    outcome = score_case(question, reference_answer, result.text)
    failure_category = None if outcome["passed"] else outcome["stage"]

    common = dict(
        model=client.model,
        config={"temperature": client.temperature, "seed": client.seed},
        prompt_version=CONFIG.prompt_version,
        dataset_version=CONFIG.dataset_version,
        evaluator_version=CONFIG.evaluator_version,
        reason=outcome.get("stage", ""),
        latency_seconds=latency,
        timestamp=time.time(),
        failure_category=failure_category,
    )

    baseline_record = BaselineRecord(
        **common,
        test_case=question,
        passed=outcome["passed"],
        score=outcome.get("similarity"),
    )
    log_entry = EvalLogEntry(
        **common,
        test_case_id=question,
        response=result.text,
        score=outcome.get("similarity"),
        passed=outcome["passed"],
    )
    return baseline_record, log_entry


def run_and_log(question: str, reference_answer: str, client: OllamaClient) -> EvalLogEntry:
    _, log_entry = run_one(question, reference_answer, client)
    log_eval(log_entry)
    return log_entry


def build_records(cases: list[dict] | None = None, client: OllamaClient | None = None) -> list[BaselineRecord]:
    cases = cases or DEFAULT_CASES
    client = client or OllamaClient()
    records = []
    for case in cases:
        record, _ = run_one(case["question"], case["reference_answer"], client)
        records.append(record)
    return records
