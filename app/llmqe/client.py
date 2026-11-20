"""The framework's one call site for talking to a model.

Started in Module 2 as a thin wrapper around `ollama.chat()`. Module 21
refactors it to delegate through the LLMProvider boundary instead of
calling `ollama.chat()` directly -- this file's public interface
(`OllamaClient.ask()` / `.ask_with_system()`) has not changed shape since
Module 2, even though every module from here on is secretly going through
a provider now.
"""

from dataclasses import dataclass

from llmqe.providers.base import LLMProvider
from llmqe.providers.ollama_provider import OllamaProvider

# Module 3's structured contract for Vespera's Helpdesk AI Assistant.
SYSTEM_PROMPT = """\
You are Vespera Helpdesk's internal product assistant, used by support \
agents to answer product questions quickly.

Respond with JSON only, using exactly these two fields:
- "answer": a plain-language answer to the agent's question, 1-3 sentences.
- "topic": exactly one of "billing", "security", "product", or "other" -- \
whichever best classifies the question.

No extra fields. No prose outside the JSON.
"""


@dataclass
class LLMResponse:
    """A captured model response, plus the settings that produced it."""

    text: str
    model: str
    temperature: float
    seed: int | None


class OllamaClient:
    """Thin wrapper around a local model call.

    Every module from Module 2 on calls this instead of talking to a
    model runtime directly, so reproducibility settings and response
    capture live in exactly one place.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        model: str = "llama3.2:3b",
        temperature: float = 0,
        seed: int | None = 42,
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._provider = provider or OllamaProvider(model, temperature, seed)

    def ask(self, prompt: str) -> LLMResponse:
        return self.ask_with_system("", prompt)

    def ask_with_system(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        text = self._provider.complete(system_prompt, user_prompt)
        return LLMResponse(
            text=text,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
        )
