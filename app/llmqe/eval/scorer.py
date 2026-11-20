"""Semantic similarity (Module 7) and the tiered `score_case()` pipeline,
refactored in Module 20 into four independently testable stages, with
Module 8's polarity-contradiction fix already applied.

Real finding from assembling this repo: this machine's local Ollama
server rejects embedding requests against `llama3.2:3b` ("This server
does not support embeddings. Start it with `--embeddings`") -- it's a
chat model, not an embedding model. Every embedding call here uses
`CONFIG.embed_model` (`nomic-embed-text`, a normal free local pull)
instead of `CONFIG.model`. Every chat/generation call in the framework
is untouched by this -- it's purely an embeddings-endpoint fix.
"""

import json
import math

from ollama import embed

from llmqe.config import CONFIG
from llmqe.eval.judge import judge_answer
from llmqe.validation import validate_response


def get_embedding(text: str, model: str = CONFIG.embed_model) -> list[float]:
    response = embed(model=model, input=text)
    return response.embeddings[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def semantic_similarity_score(answer: str, reference_answer: str) -> float:
    """Returns a 0-1 similarity score between an answer and a reference."""
    return cosine_similarity(get_embedding(answer), get_embedding(reference_answer))


def _is_polarity_contradiction(answer: str, reference_answer: str) -> bool:
    """Cheap guard for the specific false-positive Module 8 caught: a
    yes/no flip that similarity alone can't reliably detect."""
    a, r = answer.lower(), reference_answer.lower()
    a_says_no = a.strip().startswith("no") or " does not " in a or " isn't " in a
    r_says_yes = r.strip().startswith("yes") or " does " in r
    return a_says_no and r_says_yes


def _validation_stage(raw_response: str, reference_answer: str) -> dict | None:
    errors = validate_response(raw_response)
    if errors:
        return {"passed": False, "stage": "validation", "errors": errors}
    return None  # doesn't decide -- let the next stage run


def _polarity_stage(question: str, answer: str, reference_answer: str) -> dict | None:
    if not _is_polarity_contradiction(answer, reference_answer):
        return None
    verdict = judge_answer(question, reference_answer, answer)
    return {"passed": verdict["score"] >= 4, "stage": "judge (polarity guard)", "verdict": verdict}


def _similarity_stage(answer: str, reference_answer: str) -> dict | None:
    similarity = semantic_similarity_score(answer, reference_answer)
    if similarity <= 0.9:
        return None
    return {"passed": True, "stage": "similarity", "similarity": similarity}


def _judge_stage(question: str, answer: str, reference_answer: str) -> dict:
    verdict = judge_answer(question, reference_answer, answer)
    return {"passed": verdict["score"] >= 4, "stage": "judge", "verdict": verdict}


def score_case(question: str, reference_answer: str, raw_response: str) -> dict:
    validation_result = _validation_stage(raw_response, reference_answer)
    if validation_result:
        return validation_result

    answer = json.loads(raw_response)["answer"]

    for stage in (
        lambda: _polarity_stage(question, answer, reference_answer),
        lambda: _similarity_stage(answer, reference_answer),
    ):
        result = stage()
        if result:
            return result

    return _judge_stage(question, answer, reference_answer)
