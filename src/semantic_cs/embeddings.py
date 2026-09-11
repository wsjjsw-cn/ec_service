"""可插拔 embedding 后端。

背景：原先系统使用 md5 哈希词向量（hashing trick）作为"语义"向量。哈希向量只能捕捉字面
token 重叠，同义改写（如「不想要了怎么退」vs「七天无理由退货需要满足什么条件」）相似度
会塌到 0.2 左右，导致 L2 语义缓存形同虚设。

本模块提供真实语义后端，并在不可用时逐级降级，保证系统始终可运行：

    fastembed (bge-small-zh, ONNX)  ->  sentence-transformers  ->  LSA (numpy)  ->  hashing

选择优先级可用环境变量 SEMANTIC_CS_EMBEDDING_BACKEND 覆盖：
    auto | fastembed | sentence-transformers | lsa | hashing
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Iterable, Protocol

from semantic_cs.text import tokenize


class Embedder(Protocol):
    """embedding 后端统一接口。"""

    name: str
    dim: int

    def encode(self, text: str) -> list[float]:
        ...


# ============================================================
# 0. 本地 GGUF 模型（llama-cpp，优先使用，无需联网下载）
# ============================================================


class LlamaCppEmbedder:
    """加载本地 GGUF 格式的向量模型（如 bge-m3-Q8_0.gguf）。

    用户机器上已存在本地模型时优先使用：无需联网下载、无 torch 依赖、CPU 可跑。

    稳定性说明：llama.cpp 默认会占满所有 CPU 核做推理，与 httpx/asyncio 这类
    网络栈在同一进程内频繁交替调用时会出现随机段错误。这里把线程数限制为 1，
    并用互斥锁串行化 embed 调用。
    """

    name = "llama-cpp"

    def __init__(
        self,
        model_path: str = r"D:\vs_project\artical_knowledge\models\bge-m3-Q8_0.gguf",
        n_ctx: int = 8192,
        n_threads: int = 1,
    ) -> None:
        import threading

        from llama_cpp import Llama  # noqa: PLC0415

        self.model_path = model_path
        self._lock = threading.Lock()
        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
        )
        with self._lock:
            self.dim = len(self._model.embed("维度探测"))

    def encode(self, text: str) -> list[float]:
        with self._lock:
            vector = self._model.embed(text)
        return _l2_normalize([float(value) for value in vector])


# ============================================================
# 1. 哈希词向量（兜底，无依赖）
# ============================================================


class HashingEmbedder:
    """哈希词向量：零依赖、零延迟，但只能捕捉字面重叠，不具备语义泛化能力。"""

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        import hashlib

        self.dim = max(1, int(dim))
        self._hashlib = hashlib

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token, count in Counter(tokenize(text)).items():
            digest = self._hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        return _l2_normalize(vector)


# ============================================================
# 2. LSA（纯 numpy，离线语料上拟合，可捕捉共现语义）
# ============================================================


class LSAEmbedder:
    """潜在语义分析：在语料词-文档矩阵上做截断 SVD。

    相比哈希向量，LSA 能让「退款」「到账」这类共现词在向量空间靠近，
    对同义改写有一定泛化能力；代价是需要语料拟合、且无法处理语料外新词。
    """

    name = "lsa"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []
        self._u = None  # (n_terms, k)
        self._fitted = False

    def fit(self, corpus: Iterable[str]) -> "LSAEmbedder":
        import numpy as np

        docs = [tokenize(text) for text in corpus if text and text.strip()]
        if len(docs) < 2:
            # 语料太小，退化成哈希向量
            self._fitted = False
            return self

        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        self._vocab = {term: idx for idx, term in enumerate(sorted(df))}
        self._idf = [
            math.log(1.0 + len(docs) / (1.0 + df[term])) for term in self._vocab
        ]

        matrix = np.zeros((len(self._vocab), len(docs)), dtype=np.float32)
        for doc_idx, tokens in enumerate(docs):
            tf = Counter(tokens)
            for term, freq in tf.items():
                col = self._vocab.get(term)
                if col is None:
                    continue
                matrix[col, doc_idx] = (1.0 + math.log(freq)) * self._idf[col]

        k = max(2, min(self.dim, min(matrix.shape) - 1))
        u, _s, _vt = np.linalg.svd(matrix, full_matrices=False)
        self._u = u[:, :k].astype(np.float32)
        self.dim = k
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted

    def encode(self, text: str) -> list[float]:
        if not self._fitted or self._u is None:
            return HashingEmbedder(384).encode(text)
        import numpy as np

        tf = Counter(tokenize(text))
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for term, freq in tf.items():
            col = self._vocab.get(term)
            if col is None:
                continue
            vec[col] = (1.0 + math.log(freq)) * self._idf[col]
        projected = vec @ self._u
        return _l2_normalize(projected.astype(float).tolist())


# ============================================================
# 3. fastembed（bge-small-zh，ONNX 推理，无需 torch）
# ============================================================


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dim = self._read_dim()

    def _read_dim(self) -> int:
        try:
            return int(self._model._get_model_description(self.model_name)["dim"])
        except Exception:
            vector = next(iter(self._model.embed(["维度探测"])))
            return int(len(vector))

    def encode(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([text])))
        return _l2_normalize([float(value) for value in vector])


# ============================================================
# 4. sentence-transformers（效果最好，依赖 torch）
# ============================================================


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return _l2_normalize([float(value) for value in vector])


# ============================================================
# 工厂
# ============================================================

_BACKENDS: dict[str, type] = {
    "fastembed": FastEmbedEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}


def create_embedder(
    backend: str | None = None,
    dim: int = 384,
    corpus: Iterable[str] | None = None,
    model_name: str | None = None,
    local_model_path: str | None = None,
) -> Embedder:
    """按优先级创建 embedding 后端。

    backend=None 时读取环境变量 SEMANTIC_CS_EMBEDDING_BACKEND，默认 auto。
    auto 顺序：本地 GGUF(llama-cpp) -> fastembed -> sentence-transformers -> lsa -> hashing
    """
    backend = (backend or os.getenv("SEMANTIC_CS_EMBEDDING_BACKEND", "auto")).lower()
    ordered = (
        [
            "llama-cpp",
            "fastembed",
            "sentence-transformers",
            "lsa",
            "hashing",
        ]
        if backend == "auto"
        else [backend]
    )

    local_path = local_model_path or os.getenv("SEMANTIC_CS_LOCAL_EMBEDDING_MODEL")
    if not local_path:
        # 未显式指定时，回落到配置中的本地模型路径
        try:
            from semantic_cs.config import get_settings  # noqa: PLC0415

            local_path = get_settings().local_embedding_model
        except Exception:
            local_path = None

    for candidate in ordered:
        try:
            if candidate == "llama-cpp":
                if not local_path:
                    continue
                return LlamaCppEmbedder(model_path=local_path)
            if candidate == "fastembed":
                return FastEmbedEmbedder(model_name or "BAAI/bge-small-zh-v1.5")
            if candidate == "sentence-transformers":
                return SentenceTransformerEmbedder(model_name or "BAAI/bge-small-zh-v1.5")
            if candidate == "lsa":
                embedder = LSAEmbedder(dim=min(dim, 128)).fit(corpus or [])
                if embedder.fitted:
                    return embedder
                continue
            if candidate == "hashing":
                return HashingEmbedder(dim)
        except Exception as exc:  # 后端不可用时静默降级
            print(f"[Embedding] backend '{candidate}' unavailable: {exc}")

    print("[Embedding] all backends unavailable, falling back to hashing")
    return HashingEmbedder(dim)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


_CN_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")
