from __future__ import annotations

import json
from typing import Any

from semantic_cs.cache.base import SemanticCacheStore
from semantic_cs.models import CacheEntry, SemanticCandidate
from semantic_cs.text import cosine_similarity


class RedisSemanticCacheStore(SemanticCacheStore):
    """Redis Stack / RedisVL 语义缓存。

    优先使用 RedisVL 建立向量索引执行真正的 KNN 检索（HNSW，O(log n)）；
    若 Redis Stack 不支持或 RedisVL 不可用，则退化为「Redis 存向量 + Python 端暴力
    计算余弦相似度」（O(n) 全量扫描），保证功能不中断。

    旧实现的问题：``import redisvl`` 之后再没有调用过，向量以 JSON 字符串存进 hash，
    检索时全量拉回本地算 cosine —— Redis 只被当成 KV 用，完全没用到向量检索能力。
    """

    def __init__(self, redis_url: str, namespace: str = "semantic_cs:cache") -> None:
        import redis

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.namespace = namespace
        self.index_key = f"{namespace}:keys"
        self.norm_index_key = f"{namespace}:by_norm"
        self._index = None
        self._index_dim: int | None = None
        self._index_failed = False

    def _key(self, entry_key: str) -> str:
        return f"{self.namespace}:{entry_key}"

    # ============================================================
    # 向量索引（RedisVL）
    # ============================================================

    def _ensure_index(self, dim: int) -> None:
        """按向量维度创建（或复用）RedisVL 索引。"""
        if self._index_failed or (self._index is not None and self._index_dim == dim):
            return
        try:
            from redisvl.index import SearchIndex  # noqa: PLC0415

            schema = {
                "index": {
                    "name": f"{self.namespace.replace(':', '_')}_idx",
                    "prefix": f"{self.namespace}:",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "id", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": dim,
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32",
                        },
                    },
                    {"name": "entry", "type": "text"},
                    {"name": "normalized_question", "type": "text"},
                ],
            }
            index = SearchIndex.from_dict(schema, redis_client=self.client)
            index.create(overwrite=False)
            self._index = index
            self._index_dim = dim
            print(f"[Cache] RedisVL vector index ready (dims={dim})")
        except Exception as exc:  # Redis Stack 不可用 -> 回退暴力扫描
            self._index = None
            self._index_failed = True
            print(f"[Cache] RedisVL index unavailable ({exc}), fallback to brute-force scan")

    def _index_document(self, entry: CacheEntry, embedding: list[float]) -> None:
        if self._index is None:
            return
        try:
            self._index.load(
                [
                    {
                        "id": entry.key,
                        "embedding": embedding,
                        "entry": entry.model_dump_json(),
                        "normalized_question": entry.normalized_question,
                    }
                ]
            )
        except Exception as exc:
            print(f"[Cache] RedisVL load failed ({exc}), fallback to brute-force scan")
            self._index = None
            self._index_failed = True

    def _knn_search(self, embedding: list[float], top_k: int) -> list[SemanticCandidate] | None:
        if self._index is None:
            return None
        try:
            from redisvl.query import VectorQuery  # noqa: PLC0415

            query = VectorQuery(
                vector=embedding,
                vector_field_name="embedding",
                return_fields=["entry"],
                num_results=top_k,
            )
            results = self._index.query(query)
        except Exception as exc:
            print(f"[Cache] RedisVL query failed ({exc}), fallback to brute-force scan")
            self._index = None
            self._index_failed = True
            return None

        candidates: list[SemanticCandidate] = []
        for item in results:
            raw_entry = item.get("entry")
            if not raw_entry:
                continue
            distance = float(item.get("vector_distance", 0.0) or 0.0)
            candidates.append(
                SemanticCandidate(
                    entry=CacheEntry.model_validate_json(raw_entry),
                    score=max(0.0, 1.0 - distance),  # cosine distance -> similarity
                    reason="redisvl_knn",
                )
            )
        return candidates

    # ============================================================
    # 基础读写
    # ============================================================

    def upsert(self, entry: CacheEntry, embedding: list[float]) -> None:
        redis_key = self._key(entry.key)
        payload: dict[str, Any] = {
            "entry": entry.model_dump_json(),
            "embedding": json.dumps(embedding, ensure_ascii=False),
            "normalized_question": entry.normalized_question,
        }
        for field, value in payload.items():
            self.client.hset(redis_key, field, value)
        self.client.sadd(self.index_key, entry.key)
        self.client.hset(self.norm_index_key, entry.normalized_question, entry.key)

        self._ensure_index(len(embedding))
        self._index_document(entry, embedding)

    def get_by_normalized_question(self, normalized_question: str) -> CacheEntry | None:
        # 通过 Redis hash 直接定位，避免 O(n) 遍历全部条目
        entry_key = self.client.hget(self.norm_index_key, normalized_question)
        if not entry_key:
            return None
        raw = self.client.hget(self._key(str(entry_key)), "entry")
        return CacheEntry.model_validate_json(raw) if raw else None

    def all_entries(self) -> list[CacheEntry]:
        entries: list[CacheEntry] = []
        for entry_key in self.client.smembers(self.index_key):
            raw = self.client.hget(self._key(str(entry_key)), "entry")
            if raw:
                entries.append(CacheEntry.model_validate_json(raw))
        return entries

    def search(self, embedding: list[float], top_k: int = 5) -> list[SemanticCandidate]:
        knn = self._knn_search(embedding, top_k)
        if knn is not None:
            return knn

        candidates: list[SemanticCandidate] = []
        for entry_key in self.client.smembers(self.index_key):
            raw_entry = self.client.hget(self._key(str(entry_key)), "entry")
            raw_embedding = self.client.hget(self._key(str(entry_key)), "embedding")
            if not raw_entry or not raw_embedding:
                continue
            entry = CacheEntry.model_validate_json(raw_entry)
            score = cosine_similarity(embedding, json.loads(raw_embedding))
            candidates.append(SemanticCandidate(entry=entry, score=score, reason="redis_vector"))
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]

    def clear(self) -> None:
        keys = list(self.client.smembers(self.index_key))
        if keys:
            self.client.delete(*[self._key(str(key)) for key in keys])
        self.client.delete(self.index_key)
        self.client.delete(self.norm_index_key)
