from __future__ import annotations
from dataclasses import dataclass

from semantic_cs.cache.base import SemanticCacheStore
from semantic_cs.models import CacheEntry, SemanticCandidate
from semantic_cs.text import cosine_similarity


@dataclass
class _StoredEntry:
    entry: CacheEntry
    embedding: list[float]


class InMemorySemanticCacheStore(SemanticCacheStore):
    def __init__(self) -> None:
        self._entries: dict[str, _StoredEntry] = {}
        # 归一化问句 -> entry.key，让 L1 精确命中从 O(n) 全表扫描降到 O(1)
        self._by_normalized: dict[str, str] = {}

    def upsert(self, entry: CacheEntry, embedding: list[float]) -> None:
        self._entries[entry.key] = _StoredEntry(entry=entry, embedding=embedding)
        self._by_normalized[entry.normalized_question] = entry.key

    def get_by_normalized_question(self, normalized_question: str) -> CacheEntry | None:
        entry_key = self._by_normalized.get(normalized_question)
        if entry_key is None:
            return None
        stored = self._entries.get(entry_key)
        return stored.entry if stored else None

    def all_entries(self) -> list[CacheEntry]:
        return [stored.entry for stored in self._entries.values()]

    def clear(self) -> None:
        self._entries.clear()
        self._by_normalized.clear()

    def search(self, embedding: list[float], top_k: int = 5) -> list[SemanticCandidate]:
        candidates = [
            SemanticCandidate(
                entry=stored.entry,
                score=cosine_similarity(embedding, stored.embedding),
                reason="semantic_vector",
            )
            for stored in self._entries.values()
        ]
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]

