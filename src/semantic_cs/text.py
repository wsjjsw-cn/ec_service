from __future__ import annotations
import hashlib
import math
import re
import unicodedata
from collections import Counter


_PUNCT_RE = re.compile(r"[\s\t\r\n，。！？、；：,.!?;:\"'`~@#$%^&*()（）【】\[\]{}<>《》|\\/+=_-]+")
_CN_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")
_STOPWORDS = {"请问", "一下", "这个", "那个", "可以", "能不能", "怎么", "如何", "是否", "吗", "呢", "啊"}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    normalized = _PUNCT_RE.sub("", normalized)
    return normalized


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    raw_tokens = _CN_TOKEN_RE.findall(normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if token not in _STOPWORDS:
            tokens.append(token)
    chars = [tok for tok in raw_tokens if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff"]
    for size in (2, 3):
        tokens.extend("".join(chars[i : i + size]) for i in range(max(0, len(chars) - size + 1)))
    return tokens


class EmbeddingModel:
    """文本向量化门面。

    实际计算委托给 semantic_cs.embeddings 中的可插拔后端（默认为真实语义模型）。
    保留类名与构造签名以兼容既有调用点；真实维度由后端决定，dim 仅作为兜底提示。
    """

    def __init__(
        self,
        dim: int = 384,
        backend: str | None = None,
        corpus: list[str] | None = None,
        model_name: str | None = None,
        local_model_path: str | None = None,
    ) -> None:
        from semantic_cs.embeddings import create_embedder  # 延迟导入避免循环依赖

        self._backend = create_embedder(
            backend=backend,
            dim=dim,
            corpus=corpus,
            model_name=model_name,
            local_model_path=local_model_path,
        )
        self.dim = self._backend.dim
        # 同一个问句在一次请求内会被编码多次（L2 召回 + 检索补全），
        # 真实模型单次编码约数十毫秒，缓存后可显著降低延迟
        self._cache: dict[str, list[float]] = {}
        self._cache_size = 2048

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def encode(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = self._backend.encode(text)
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[text] = vector
        return vector

    def clear_cache(self) -> None:
        """清空编码缓存（离线评测需要逐轮冷启动，否则先跑的轮次会给后跑的轮次预热）。"""
        self._cache.clear()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    denom = left_norm * right_norm
    return numerator / denom if denom else 0.0


def edit_ratio(left: str, right: str) -> float:
    if left == right:
        return 100.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        current = [i]
        for j, rc in enumerate(right, start=1):
            cost = 0 if lc == rc else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    distance = previous[-1]
    return (1 - distance / max(len(left), len(right))) * 100

