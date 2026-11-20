"""The minimal provider boundary (Module 21).

One method, one job: turn a system/user prompt pair into a raw text
completion. Nothing else in this package is allowed to know which model
runtime is actually behind it.
"""

from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion for a system/user prompt pair."""
        ...
