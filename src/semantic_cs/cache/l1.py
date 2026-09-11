from __future__ import annotations

from semantic_cs.cache.base import SemanticCacheStore
from semantic_cs.models import SemanticCandidate
from semantic_cs.text import edit_ratio, normalize_text, tokenize


class L1RuleCache:
    def __init__(self, store: SemanticCacheStore, edit_threshold: float = 82.0, overlap_threshold: float = 0.65) -> None:
        self.store = store
        self.edit_threshold = edit_threshold
        self.overlap_threshold = overlap_threshold
        # 缓存条目的分词结果：lookup 会对全表做编辑距离与重叠度计算，
        # 不缓存的话每次请求都要把全部条目的 question 重新分词一遍
        self._token_cache: dict[str, set[str]] = {}

    def _entry_tokens(self, entry) -> set[str]:
        cached = self._token_cache.get(entry.key)
        if cached is None:
            cached = set(tokenize(entry.question))
            self._token_cache[entry.key] = cached
        return cached

    def lookup(self, question: str) -> SemanticCandidate | None:
        normalized = normalize_text(question)
        exact = self.store.get_by_normalized_question(normalized)
        if exact:
            return SemanticCandidate(entry=exact, score=1.0, reason="l1_exact_normalized")

        intent = self._intent_shortcut(question)
        if intent:
            return intent

        best: SemanticCandidate | None = None
        question_tokens = set(tokenize(question))
        for entry in self.store.all_entries():
            ratio = edit_ratio(normalized, entry.normalized_question)
            overlap = self._subquestion_overlap(question_tokens, self._entry_tokens(entry))
            score = max(ratio / 100.0, overlap)
            if ratio >= self.edit_threshold:
                candidate = SemanticCandidate(entry=entry, score=score, reason="l1_edit_distance")
            elif overlap >= self.overlap_threshold and len(question_tokens) >= 3:
                candidate = SemanticCandidate(entry=entry, score=score, reason="l1_subquestion")
            else:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _intent_shortcut(self, question: str) -> SemanticCandidate | None:
        normalized = normalize_text(question)
        rules = [
            ({"退款"}, {"到账", "多长时间", "多久"}, {"退款", "到账"}),
            ({"发票"}, {"申请", "操作", "开"}, {"发票", "申请"}),
        ]
        for must_terms, optional_terms, entry_terms in rules:
            if not all(term in normalized for term in must_terms):
                continue
            if not any(term in normalized for term in optional_terms):
                continue
            for entry in self.store.all_entries():
                entry_text = normalize_text(entry.question)
                if all(term in entry_text for term in entry_terms):
                    return SemanticCandidate(entry=entry, score=0.98, reason="l1_intent_shortcut")
        return None

    @staticmethod
    def _subquestion_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, min(len(left), len(right)))

