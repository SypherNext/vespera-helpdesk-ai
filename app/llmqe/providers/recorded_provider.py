"""A fake provider that replays a fixed response (Module 21).

Proves the OllamaClient -> LLMProvider boundary is real: calling code can
be tested without a live Ollama server at all.
"""


class RecordedProvider:
    """Replays a fixed response instead of calling a model. Useful for
    testing calling code's behavior without depending on a live model."""

    def __init__(self, response_text: str):
        self._response_text = response_text

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response_text
