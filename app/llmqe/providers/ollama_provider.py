"""The real provider: talks to the local Ollama server."""

from ollama import chat

from llmqe.config import CONFIG


class OllamaProvider:
    def __init__(
        self,
        model: str = CONFIG.model,
        temperature: float = CONFIG.temperature,
        seed: int | None = CONFIG.seed,
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "seed": self.seed},
        )
        return response.message.content
