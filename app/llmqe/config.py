"""Single source of configuration for the framework.

Module 20 ("Clean Code & Design Principles for the Framework") introduces
this file to stop the same literals ("llama3.2:3b", "v1", the safety
block-category set) from being duplicated -- and silently drifting out of
sync -- across baseline.py, logger.py, and regression_suite.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameworkConfig:
    model: str = "llama3.2:3b"
    # Real finding while assembling this repo: this machine's Ollama server
    # returns "This server does not support embeddings" for llama3.2:3b --
    # it is a chat model, not an embedding model. nomic-embed-text (already
    # a normal, free, local `ollama pull`) is used for every embedding call
    # instead. Every chat/generation call in this framework still uses
    # `model` (llama3.2:3b) -- only embedding calls use `embed_model`.
    embed_model: str = "nomic-embed-text"
    temperature: float = 0
    seed: int | None = 42
    prompt_version: str = "v1"
    dataset_version: str = "v1"
    evaluator_version: str = "v1"
    block_categories: frozenset = frozenset({"safety_failure"})
    latency_budget_seconds: float = 3.0


CONFIG = FrameworkConfig()
