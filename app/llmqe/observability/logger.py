"""Module 17: a structured, permanent record of every evaluation run, so
a future failure gets diagnosed by reading a log instead of re-running ad
hoc print() statements.

Never logs credentials, API keys, secrets, PII, or real production
data -- synthetic data only, same rule the whole course follows since
Module 5's fabricated ticket data.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

LOG_PATH = Path(__file__).parent / "eval_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class EvalLogEntry:
    model: str
    config: dict
    prompt_version: str
    dataset_version: str
    evaluator_version: str
    test_case_id: str
    response: str
    score: float | None
    reason: str
    latency_seconds: float
    timestamp: float
    passed: bool
    failure_category: str | None


def log_eval(entry: EvalLogEntry) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f]
