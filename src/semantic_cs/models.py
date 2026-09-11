from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeVar


class QueryRoute(str, Enum):
    DYNAMIC = "dynamic"
    CACHE = "cache"
    RAG = "rag"
    IDENTITY = "identity"
    CHITCHAT = "chitchat"
    OOD = "ood"


class CacheDecision(str, Enum):
    REUSE = "reuse"
    COMPLETE = "complete"
    RAG = "rag"


T = TypeVar("T", bound="ModelMixin")


class ModelMixin:
    @classmethod
    def model_validate(cls: type[T], payload: dict[str, Any]) -> T:
        return cls(**payload)

    @classmethod
    def model_validate_json(cls: type[T], payload: str) -> T:
        return cls.model_validate(json.loads(payload))

    def model_dump_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def model_copy(self: T, update: dict[str, Any] | None = None) -> T:
        data = asdict(self)
        data.update(update or {})
        return self.__class__(**data)


@dataclass
class FAQItem(ModelMixin):
    id: str
    question: str
    answer: str
    tags: list[str] = field(default_factory=list)
    hot: bool = False


@dataclass
class KnowledgeDoc(ModelMixin):
    id: str
    title: str
    body: str


@dataclass
class Product(ModelMixin):
    sku: str
    name: str
    stock: int
    price: float
    updated_at: str


@dataclass
class CacheEntry(ModelMixin):
    key: str
    question: str
    normalized_question: str
    answer: str
    source: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCandidate(ModelMixin):
    entry: CacheEntry
    score: float
    reason: str = ""


@dataclass
class RetrievalHit(ModelMixin):
    doc: KnowledgeDoc
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0


@dataclass
class ChatRequest(ModelMixin):
    question: str
    user_id: str | None = None


@dataclass
class ChatResponse(ModelMixin):
    answer: str
    route: QueryRoute
    latency_ms: float
    llm_calls: int
    cache_level: str | None = None
    decision: CacheDecision | None = None
    sources: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult(ModelMixin):
    decision: CacheDecision
    candidate: SemanticCandidate | None
    missing_terms: list[str] = field(default_factory=list)
    confidence: float = 0.0

