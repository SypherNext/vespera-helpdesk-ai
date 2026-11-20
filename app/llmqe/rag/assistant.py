"""Module 9/11: Vespera's assistant, now RAG-backed. Module 11 adds the
`source_doc_id` citation field to the contract."""

from llmqe.client import OllamaClient
from llmqe.rag.retriever import Retriever

RAG_SYSTEM_PROMPT = """\
You are Vespera Helpdesk's internal product assistant. You will be given \
retrieved documentation (each piece labeled with a source id) and a \
question. Answer using ONLY the retrieved documentation -- if it doesn't \
contain the answer, say so explicitly instead of guessing.

Respond with JSON only:
{"answer": "...", "topic": "billing|security|product|other", "source_doc_id": "<id of the document the answer came from, or null>"}
"""


def ask_with_retrieval(question: str, retriever: Retriever, client: OllamaClient) -> dict:
    results = retriever.retrieve(question)
    context = "\n".join(f"[{r['document']['id']}] {r['document']['text']}" for r in results)
    prompt = f"Retrieved documentation:\n{context}\n\nQuestion: {question}"
    result = client.ask_with_system(RAG_SYSTEM_PROMPT, prompt)
    return {"response": result, "retrieved": results}
