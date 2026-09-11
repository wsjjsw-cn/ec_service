from __future__ import annotations
from semantic_cs.cache.base import SemanticCacheStore
from semantic_cs.models import SemanticCandidate
from semantic_cs.text import EmbeddingModel


class L2SemanticCache:
    def __init__(
        self,
        store: SemanticCacheStore,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.78,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def lookup(self, question: str, top_k: int = 3) -> list[SemanticCandidate]:
        embedding = self.embedding_model.encode(question)
        candidates = self.store.search(embedding, top_k=top_k)
        return [candidate for candidate in candidates if candidate.score >= self.similarity_threshold]

