from __future__ import annotations

from semantic_cs.cache.base import SemanticCacheStore
from semantic_cs.cache.l1 import L1RuleCache
from semantic_cs.cache.l2 import L2SemanticCache
from semantic_cs.cache.memory import InMemorySemanticCacheStore
from semantic_cs.cache.redis_store import RedisSemanticCacheStore
from semantic_cs.config import Settings, get_settings
from semantic_cs.data_loader import load_docs, load_faqs, load_products
from semantic_cs.generator import AnswerGenerator
from semantic_cs.guardrails import DynamicQueryGuardrail
from semantic_cs.validator import LLMValidator
from semantic_cs.langchain_setup import create_llm
from semantic_cs.langgraph_agent import LangGraphReActAgent
from semantic_cs.llm_usage import TokenUsageTracker
from semantic_cs.models import CacheEntry
from semantic_cs.pipeline import CustomerServicePipeline
from semantic_cs.retrieval import HybridRetriever
from semantic_cs.text import EmbeddingModel, normalize_text


def create_store(settings: Settings | None = None) -> SemanticCacheStore:
    settings = settings or get_settings()
    if settings.use_redis:
        try:
            store = RedisSemanticCacheStore(settings.redis_url)
            store.client.ping()
            print("[Cache] Redis connection successful")
            return store
        except Exception as e:
            print(f"[Cache] Redis not available ({e}), using in-memory store")
            return InMemorySemanticCacheStore()
    print("[Cache] Using in-memory store")
    return InMemorySemanticCacheStore()


def preload_hot_faqs(store: SemanticCacheStore, embedding_model: EmbeddingModel | None = None) -> int:
    settings = get_settings()
    if embedding_model is None:
        faq_corpus = [f"{faq.question} {faq.answer}" for faq in load_faqs()]
        embedding_model = EmbeddingModel(
            settings.vector_dim,
            backend=settings.embedding_backend,
            corpus=faq_corpus,
            model_name=settings.embedding_model_name,
            local_model_path=settings.local_embedding_model,
        )
    count = 0
    faqs = load_faqs()
    for faq in faqs:
        if not faq.hot:
            continue
        entry = CacheEntry(
            key=f"faq:{faq.id}",
            question=faq.question,
            normalized_question=normalize_text(faq.question),
            answer=faq.answer,
            source=faq.id,
            tags=faq.tags,
            metadata={"preloaded": True},
        )
        store.upsert(entry, embedding_model.encode(faq.question))
        count += 1
    print(f"[Cache] Preloaded {count}/{len(faqs)} hot FAQs")
    return count


def _build_callback_manager(tracker: TokenUsageTracker):
    """把 tracker 包装成 CallbackManager。

    langchain-core 1.x 的模型构造函数要求 callbacks 是 BaseCallbackManager 实例，
    直接传 [handler] 会被 pydantic 校验拒绝。构造失败会让整个系统静默降级到无 LLM。
    """
    try:
        from langchain_core.callbacks import CallbackManager

        return CallbackManager([tracker])
    except Exception:
        return [tracker]


def build_pipeline(preload: bool = True, settings: Settings | None = None) -> CustomerServicePipeline:
    """构建完整的 Pipeline"""
    settings = settings or get_settings()
    
    print("\n" + "=" * 60)
    print("初始化智能客服系统 (LangChain + LangGraph)")
    print("=" * 60)
    
    # 1. 加载知识库语料（LSA 后端需要先拟合，故先于 embedding 构建）
    docs = load_docs()
    products = load_products()
    faqs = load_faqs()
    corpus = [f"{doc.title} {doc.body}" for doc in docs] + [
        f"{faq.question} {faq.answer}" for faq in faqs
    ]

    # 2. 初始化 Embedding（真实语义模型优先，见 semantic_cs.embeddings）
    embedding_model = EmbeddingModel(
        settings.vector_dim,
        backend=settings.embedding_backend,
        corpus=corpus,
        model_name=settings.embedding_model_name,
        local_model_path=settings.local_embedding_model,
    )
    print(f"[Embedding] backend={embedding_model.backend_name} dim={embedding_model.dim}")

    # 3. 初始化缓存存储
    store = create_store(settings)

    # 4. 初始化 LLM（挂载 token 用量追踪，用于按真实 token 计量成本）
    llm_tracker = TokenUsageTracker()
    llm = None
    if settings.use_real_llm:
        llm = create_llm(
            model=settings.llm_model,
            api_key=settings.llm_api_key or None,
            temperature=0.7,
            max_tokens=settings.llm_max_tokens,
            callbacks=_build_callback_manager(llm_tracker),
        )
    use_llm = llm is not None
    print(f"[LLM] Enabled: {use_llm}")

    # 5. 预加载 FAQs
    if preload:
        preload_hot_faqs(store, embedding_model)

    print(f"[Knowledge] Loaded {len(docs)} documents, {len(products)} products")
    
    # 6. 初始化各个组件
    retriever = HybridRetriever(docs, embedding_model)
    l1_cache = L1RuleCache(store, settings.l1_edit_threshold, settings.l1_overlap_threshold)
    l2_cache = L2SemanticCache(store, embedding_model, settings.l2_similarity_threshold)
    validator = LLMValidator(
        settings.validator_reuse_threshold,
        settings.validator_complete_threshold,
        llm=llm,
        use_llm=use_llm,
    )
    generator = AnswerGenerator(llm=llm, use_llm=use_llm, max_memory_messages=10)
    agent = LangGraphReActAgent(
        llm=llm,
        retriever=retriever,
        top_k=settings.retrieval_top_k,
        use_llm=use_llm,
        max_memory_messages=10,
        # 不重复挂回调：llm 构造时已挂 CallbackManager，
        # 再往 graph.invoke 传 handler list 会被当成 CallbackManager 使用而报错
    )
    
    # 7. 初始化 Pipeline
    pipeline = CustomerServicePipeline(
        store=store,
        embedding_model=embedding_model,
        guardrail=DynamicQueryGuardrail(products),
        l1_cache=l1_cache,
        l2_cache=l2_cache,
        validator=validator,
        retriever=retriever,
        generator=generator,
        agent=agent,
        llm_tracker=llm_tracker,
    )
    
    print(f"[Pipeline] Ready!")
    print("=" * 60 + "\n")
    
    return pipeline
