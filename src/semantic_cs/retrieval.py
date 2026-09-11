from __future__ import annotations
import math
from collections import Counter, defaultdict

from semantic_cs.models import KnowledgeDoc, RetrievalHit
from semantic_cs.text import EmbeddingModel, cosine_similarity, tokenize


class HybridRetriever:
    def __init__(
        self,
        docs: list[KnowledgeDoc],
        embedding_model: EmbeddingModel,
        vector_weight: float = 0.55,
        keyword_weight: float = 0.45,
    ) -> None:
        self.docs = docs
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.doc_embeddings = {doc.id: embedding_model.encode(f"{doc.title} {doc.body}") for doc in docs}
        self.doc_tokens = {doc.id: tokenize(f"{doc.title} {doc.body}") for doc in docs}
        self.idf = self._build_idf()

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        query_embedding = self.embedding_model.encode(query)
        query_terms = tokenize(query)
        raw_hits: list[RetrievalHit] = []
        for doc in self.docs:
            vector_score = max(0.0, cosine_similarity(query_embedding, self.doc_embeddings[doc.id]))
            keyword_score = self._bm25(query_terms, self.doc_tokens[doc.id])
            raw_hits.append(RetrievalHit(doc=doc, score=0.0, vector_score=vector_score, keyword_score=keyword_score))

        max_keyword = max((hit.keyword_score for hit in raw_hits), default=0.0) or 1.0
        hits: list[RetrievalHit] = []
        for hit in raw_hits:
            normalized_keyword = hit.keyword_score / max_keyword
            score = self.vector_weight * hit.vector_score + self.keyword_weight * normalized_keyword
            hits.append(
                RetrievalHit(
                    doc=hit.doc,
                    score=score,
                    vector_score=hit.vector_score,
                    keyword_score=normalized_keyword,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self.docs)
        df: defaultdict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens.values():
            for token in set(tokens):
                df[token] += 1
        return {token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}

    def _bm25(self, query_terms: list[str], doc_terms: list[str]) -> float:
        if not query_terms or not doc_terms:
            return 0.0
        k1 = 1.5
        b = 0.75
        avgdl = sum(len(tokens) for tokens in self.doc_tokens.values()) / max(1, len(self.doc_tokens))
        tf = Counter(doc_terms)
        score = 0.0
        doc_len = len(doc_terms)
        for term in query_terms:
            freq = tf[term]
            if freq == 0:
                continue
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / avgdl)
            score += self.idf.get(term, 0.0) * numerator / denominator
        return score

