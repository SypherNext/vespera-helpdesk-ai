"""Module 9: a small retriever over Vespera's documentation.

Pre-computes every document's embedding once at construction, rather
than re-embedding the whole knowledge base on every query -- Module 16
diagnoses exactly what happens when a caller forgets this and builds a
fresh Retriever() on every single request instead of reusing one.
"""

from llmqe.eval.scorer import cosine_similarity, get_embedding
from llmqe.rag.documents import DOCUMENTS


class Retriever:
    def __init__(self, documents=DOCUMENTS):
        self.documents = documents
        self._doc_embeddings = [(doc, get_embedding(doc["text"])) for doc in documents]

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        query_embedding = get_embedding(query)
        scored = [
            (doc, cosine_similarity(query_embedding, doc_embedding))
            for doc, doc_embedding in self._doc_embeddings
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [{"document": doc, "score": score} for doc, score in scored[:top_k]]
