"""Setup lesson's three-line confirmation script: if this prints a short
greeting, your Ollama + Python environment is ready for the rest of the
course."""

from ollama import chat

response = chat(model="llama3.2:3b", messages=[{"role": "user", "content": "Say hello in five words or fewer."}])
print(response.message.content)
