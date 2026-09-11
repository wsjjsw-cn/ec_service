from __future__ import annotations
import time
import uuid

from semantic_cs.cache.base import SemanticCacheStore
from semantic_cs.cache.l1 import L1RuleCache
from semantic_cs.cache.l2 import L2SemanticCache
from semantic_cs.generator import AnswerGenerator
from semantic_cs.guardrails import DynamicQueryGuardrail
from semantic_cs.llm_usage import TokenUsageTracker
from semantic_cs.intent_classifier import IntentClassifier, IntentType
from semantic_cs.models import CacheDecision, CacheEntry, ChatResponse, QueryRoute
from semantic_cs.langgraph_agent import LangGraphReActAgent
from semantic_cs.retrieval import HybridRetriever
from semantic_cs.text import EmbeddingModel, normalize_text
from semantic_cs.validator import LLMValidator


class CustomerServicePipeline:
    def __init__(
        self,
        store: SemanticCacheStore,
        embedding_model: EmbeddingModel,
        guardrail: DynamicQueryGuardrail,
        l1_cache: L1RuleCache,
        l2_cache: L2SemanticCache,
        validator: LLMValidator,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        agent: LangGraphReActAgent,
        intent_classifier: IntentClassifier | None = None,
        llm_tracker: "TokenUsageTracker | None" = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.guardrail = guardrail
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.validator = validator
        self.retriever = retriever
        self.generator = generator
        self.agent = agent
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.llm_tracker = llm_tracker

    def ask(
        self,
        question: str,
        session_id: str = "default",
        write_back: bool = True,
        use_memory: bool = True,
        bypass_cache: bool = False,
    ) -> ChatResponse:
        """处理用户提问。

        bypass_cache=True 时跳过动态拦截、L1/L2 与 Validator，直接走完整 RAG，
        用于离线评测构造"无缓存中间件"的真实基线。该模式不会写回缓存。
        """
        started = time.perf_counter()
        validator_calls = 0

        if not bypass_cache:
            # 1. Guardrail: 检查是否是动态查询
            guardrail_result = self.guardrail.inspect(question)
            if guardrail_result.is_dynamic:
                answer = self.guardrail.answer_dynamic(question, guardrail_result)
                return self._response(
                    started,
                    answer,
                    QueryRoute.DYNAMIC,
                    llm_calls=0,
                    debug={"guardrail": guardrail_result.reason}
                )

            # 2. 意图分类
            intent_result = self.intent_classifier.classify(question)

            # 2.1 身份介绍
            if intent_result.intent == IntentType.IDENTITY:
                answer = self.generator.generate_identity_response(question)
                return self._response(
                    started,
                    answer,
                    QueryRoute.IDENTITY,
                    llm_calls=0,
                    debug={"intent": "identity", "reason": intent_result.reason}
                )

            # 2.2 闲聊
            if intent_result.intent == IntentType.CHITCHAT:
                answer = self.generator.generate_chitchat_response(question)
                return self._response(
                    started,
                    answer,
                    QueryRoute.CHITCHAT,
                    llm_calls=0,
                    debug={"intent": "chitchat", "reason": intent_result.reason}
                )

            # 2.3 超出域
            if intent_result.intent == IntentType.OUT_OF_DOMAIN:
                answer = self.generator.generate_out_of_domain_response(question)
                return self._response(
                    started,
                    answer,
                    QueryRoute.OOD,
                    llm_calls=1,
                    debug={"intent": "out_of_domain", "reason": intent_result.reason}
                )

            # 3. L1 Cache: 规则匹配
            l1_candidate = self.l1_cache.lookup(question)
            if l1_candidate:
                return self._response(
                    started,
                    l1_candidate.entry.answer,
                    QueryRoute.CACHE,
                    cache_level="L1",
                    decision=CacheDecision.REUSE,
                    sources=[l1_candidate.entry.source],
                    llm_calls=0,
                    debug={"reason": l1_candidate.reason, "score": l1_candidate.score},
                )

            # 4. L2 Cache: 语义匹配 + Validator
            l2_candidates = self.l2_cache.lookup(question)
            validation = self.validator.validate(question, l2_candidates)
            validator_calls = 1 if l2_candidates and self.validator.use_llm else 0

            # 4.1 直接复用
            if validation.decision == CacheDecision.REUSE and validation.candidate:
                return self._response(
                    started,
                    validation.candidate.entry.answer,
                    QueryRoute.CACHE,
                    cache_level="L2",
                    decision=CacheDecision.REUSE,
                    sources=[validation.candidate.entry.source],
                    llm_calls=validator_calls,
                    debug={"score": validation.confidence, "reason": validation.candidate.reason},
                )

            # 4.2 补全复用
            if validation.decision == CacheDecision.COMPLETE and validation.candidate:
                hits = self.retriever.retrieve(question, top_k=3)
                answer = self.generator.complete_from_cache(
                    question,
                    validation.candidate.entry.answer,
                    hits,
                    session_id=session_id,
                    use_memory=use_memory,
                )
                if write_back:
                    self._write_cache(question, answer, "validator_complete", [hit.doc.id for hit in hits])
                return self._response(
                    started,
                    answer,
                    QueryRoute.CACHE,
                    cache_level="L2+RAG",
                    decision=CacheDecision.COMPLETE,
                    sources=[validation.candidate.entry.source, *[hit.doc.id for hit in hits]],
                    llm_calls=validator_calls + 1,
                    debug={"missing_terms": validation.missing_terms, "score": validation.confidence},
                )

        # 5. 完整 RAG
        answer, hits, trace, rag_calls = self.agent.answer(
            question,
            session_id=session_id,
            use_memory=use_memory,
        )

        # 5.1 低置信度 -> 返回 OOD
        if self.generator.is_low_confidence(hits):
            return self._response(
                started,
                self.generator.generate_out_of_domain_response(question),
                QueryRoute.OOD,
                llm_calls=rag_calls + 1,
                debug={"reason": "low_confidence", "top_score": hits[0].score if hits else 0},
            )

        # 5.2 写回缓存
        if write_back and self._should_cache(question, hits):
            self._write_cache(question, answer, "rag_write_back", [hit.doc.id for hit in hits])

        # 5.3 返回 RAG 结果
        debug_info = trace if isinstance(trace, dict) else {}
        return self._response(
            started,
            answer,
            QueryRoute.RAG,
            decision=CacheDecision.RAG,
            sources=[hit.doc.id for hit in hits],
            llm_calls=validator_calls + rag_calls,
            debug=debug_info,
        )

    # ============================================================
    # 辅助方法
    # ============================================================

    def _write_cache(self, question: str, answer: str, source: str, source_docs: list[str]) -> None:
        """写回缓存"""
        entry = CacheEntry(
            key=f"qa:{uuid.uuid5(uuid.NAMESPACE_URL, normalize_text(question))}",
            question=question,
            normalized_question=normalize_text(question),
            answer=answer,
            source=source,
            metadata={"source_docs": source_docs},
        )
        self.store.upsert(entry, self.embedding_model.encode(question))

    @staticmethod
    def _should_cache(question: str, hits: list) -> bool:
        """判断是否应该缓存"""
        return bool(hits) and hits[0].score >= 0.35 and len(question) <= 80

    @staticmethod
    def _response(
        started: float,
        answer: str,
        route: QueryRoute,
        cache_level: str | None = None,
        decision: CacheDecision | None = None,
        sources: list[str] | None = None,
        llm_calls: int = 0,
        debug: dict | None = None,
    ) -> ChatResponse:
        """创建响应"""
        return ChatResponse(
            answer=answer,
            route=route,
            cache_level=cache_level,
            decision=decision,
            sources=sources or [],
            latency_ms=(time.perf_counter() - started) * 1000,
            llm_calls=llm_calls,
            debug=debug or {},
        )

    # ============================================================
    # 记忆管理
    # ============================================================

    def get_session_memory(self, session_id: str = "default") -> list:
        """获取会话记忆"""
        return self.generator.get_memory(session_id)

    def clear_session_memory(self, session_id: str = "default") -> None:
        """清除会话记忆"""
        self.generator.clear_memory(session_id)
        self.agent.clear_memory(session_id)

    def clear_all_memory(self) -> None:
        """清除所有记忆"""
        self.generator.clear_all_memory()
        self.agent.clear_all_memory()
