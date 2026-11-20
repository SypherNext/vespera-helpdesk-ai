"""LLM-as-a-judge (Module 7), with Module 23's fix applied: the judge's
own OllamaClient pins its settings from CONFIG instead of a local literal,
so a stray experimental edit can't silently drift the judge's temperature
away from 0 and make evaluation results flaky for reasons that have
nothing to do with the system under test.
"""

import json

from llmqe.client import OllamaClient
from llmqe.config import CONFIG

JUDGE_SYSTEM_PROMPT = """\
You are an evaluator grading a support assistant's answer. You will be \
given a question, a reference answer known to be correct, and a candidate \
answer to grade.

Score the candidate from 1 to 5:
5 = fully correct and consistent with the reference
3 = partially correct, missing or vague on a key detail
1 = contradicts the reference or states something false

Respond with JSON only: {"score": <1-5>, "reason": "<one sentence>"}
"""


def judge_answer(question: str, reference_answer: str, candidate_answer: str) -> dict:
    client = OllamaClient(temperature=0, seed=CONFIG.seed)
    prompt = (
        f"Question: {question}\n"
        f"Reference answer: {reference_answer}\n"
        f"Candidate answer: {candidate_answer}"
    )
    result = client.ask_with_system(JUDGE_SYSTEM_PROMPT, prompt)
    return json.loads(result.text)
