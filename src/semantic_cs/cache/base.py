from __future__ import annotations
from abc import ABC, abstractmethod

from semantic_cs.models import CacheEntry, SemanticCandidate


class SemanticCacheStore(ABC):
    @abstractmethod
    def upsert(self, entry: CacheEntry, embedding: list[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_by_normalized_question(self, normalized_question: str) -> CacheEntry | None:
        raise NotImplementedError

    @abstractmethod
    def all_entries(self) -> list[CacheEntry]:
        raise NotImplementedError

    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 5) -> list[SemanticCandidate]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """清空全部缓存条目（评测重置用）。"""
        raise NotImplementedError

