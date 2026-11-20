"""Module 21: proves the provider boundary is real -- this test needs no
Ollama server at all."""

from llmqe.client import OllamaClient
from llmqe.providers.recorded_provider import RecordedProvider
from llmqe.validation import validate_response


def test_validate_response_handles_malformed_json():
    fake_provider = RecordedProvider(response_text="not valid json at all")
    client = OllamaClient(provider=fake_provider)
    result = client.ask_with_system("system", "does the pro plan support sso?")

    errors = validate_response(result.text)
    assert errors == ["response is not valid JSON"]
