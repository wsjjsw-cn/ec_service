import os

from semantic_cs.bootstrap import build_pipeline
from semantic_cs.config import Settings
from semantic_cs.data_loader import load_eval_requests
from semantic_cs.evaluation import OfflineEvaluator
from semantic_cs.models import QueryRoute

# 常规用例关闭真实 LLM：否则每个用例都会打 API，既慢又消耗额度。
# 需要验证 LLM 链路的用例见文件末尾「LLM 集成」分组。
_REDIS_OFF = Settings(use_redis=False, use_real_llm=False)

# LLM 集成用例的开关：默认开启，设置 SEMANTIC_CS_SKIP_LLM_TESTS=1 可跳过
_SKIP_LLM_TESTS = os.getenv("SEMANTIC_CS_SKIP_LLM_TESTS", "0") == "1"


def _semantic_backend_enabled() -> bool:
    """语义测试依赖本地 bge-m3 模型；模型缺失时系统会降级到哈希向量，本组用例跳过。"""
    return build_pipeline(settings=_REDIS_OFF).embedding_model.backend_name != "hashing"


def test_dynamic_product_stock_is_intercepted() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("AeroPods Pro 2 现在还有库存吗？")
    assert response.route == QueryRoute.DYNAMIC
    assert "库存" in response.answer
    assert response.llm_calls == 0


def test_l1_cache_reuses_preloaded_faq() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("七天无理由退货需要满足什么条件")
    assert response.route == QueryRoute.CACHE
    assert response.cache_level == "L1"
    assert "不影响二次销售" in response.answer


def test_rag_writes_back_then_l1_hits() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    first = pipeline.ask("人为损坏可以免费维修吗？")
    second = pipeline.ask("人为损坏可以免费维修吗")
    assert first.route == QueryRoute.RAG
    assert second.route == QueryRoute.CACHE


def test_identity_intent_returns_introduction() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("你是谁")
    assert response.route == QueryRoute.IDENTITY
    assert "客服" in response.answer
    assert response.llm_calls == 0


def test_identity_variants_return_introduction() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    test_cases = ["你是啥", "你是什么", "你是做什么的"]
    for question in test_cases:
        response = pipeline.ask(question)
        assert response.route == QueryRoute.IDENTITY, f"'{question}' should be IDENTITY but got {response.route}"


def test_chitchat_greeting_returns_friendly_response() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("你好")
    assert response.route == QueryRoute.CHITCHAT
    assert "客服" in response.answer
    assert response.llm_calls == 0


def test_chitchat_thanks_returns_polite_response() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("谢谢")
    assert response.route == QueryRoute.CHITCHAT
    assert response.llm_calls == 0


def test_chitchat_farewell_returns_polite_response() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("再见")
    assert response.route == QueryRoute.CHITCHAT
    assert response.llm_calls == 0


def test_out_of_domain_returns_polite_refusal() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("讲个笑话")
    assert response.route == QueryRoute.OOD
    assert response.llm_calls >= 0  # OOD now uses LLM when available


def test_out_of_domain_identity_question() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("推荐一部电影")
    assert response.route == QueryRoute.OOD
    assert response.llm_calls >= 0  # OOD now uses LLM when available


def test_business_question_not_classified_as_out_of_domain() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("保修政策是什么")
    assert response.route != QueryRoute.OOD
    assert response.route != QueryRoute.IDENTITY
    assert response.route != QueryRoute.CHITCHAT


def test_business_question_not_classified_as_identity() -> None:
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("如何维修商品")
    assert response.route != QueryRoute.IDENTITY


# ============================================================
# 回归测试：动态拦截完整性
# ============================================================


def test_presale_shipping_time_is_intercepted() -> None:
    """预售发货时间没有商品名，旧实现会因缺少商品匹配而漏拦截。"""
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("预售商品什么时候发货？")
    assert response.route == QueryRoute.DYNAMIC
    assert response.llm_calls == 0


def test_all_dynamic_samples_are_intercepted() -> None:
    """评测集中标注为 dynamic 的样本必须全部被拦截。"""
    pipeline = build_pipeline(settings=_REDIS_OFF)
    dynamic_samples = [
        item["question"] for item in load_eval_requests() if item.get("kind") == "dynamic"
    ]
    assert dynamic_samples, "评测集应包含动态样本"
    for question in dynamic_samples:
        response = pipeline.ask(question)
        assert response.route == QueryRoute.DYNAMIC, f"'{question}' 应被拦截"


# ============================================================
# 回归测试：意图分类不应把业务问题判为域外
# ============================================================


def test_invoice_detail_question_not_out_of_domain() -> None:
    """「增值税专票」不含「发票」二字，旧关键词表会误判为域外。"""
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("增值税专票需要填写什么？")
    assert response.route != QueryRoute.OOD


def test_points_question_not_out_of_domain() -> None:
    """「积分可以提现吗」不含「会员」二字，旧关键词表会误判为域外。"""
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("积分可以提现吗？")
    assert response.route != QueryRoute.OOD


# ============================================================
# 回归测试：Validator 返回类型
# ============================================================


def test_validator_returns_semantic_candidate() -> None:
    """Validator 必须返回 SemanticCandidate，pipeline 按 .entry.answer 解包。

    旧实现返回 CacheEntry，一旦 L2 产生候选就会 AttributeError；
    哈希向量下 L2 从不产生候选，该分支从未被执行，故一直没暴露。
    """
    pipeline = build_pipeline(settings=_REDIS_OFF)
    entry = next(iter(pipeline.store.all_entries()))
    from semantic_cs.models import SemanticCandidate

    candidate = SemanticCandidate(entry=entry, score=0.95, reason="test")
    result = pipeline.validator.validate(entry.question, [candidate])
    assert result.candidate is not None
    assert hasattr(result.candidate, "entry"), "candidate 必须是 SemanticCandidate"


# ============================================================
# 回归测试：评测可复现
# ============================================================


_EVAL_OFFLINE = Settings(use_redis=False, use_real_llm=False)


def test_offline_eval_is_idempotent() -> None:
    """重复调用 /metrics/eval 不应因缓存回写而让命中率一轮比一轮高。"""
    first = OfflineEvaluator(settings=_EVAL_OFFLINE).run()
    second = OfflineEvaluator(settings=_EVAL_OFFLINE).run()
    assert first.cache_coverage_rate == second.cache_coverage_rate
    assert first.direct_cache_reuse_rate == second.direct_cache_reuse_rate


def test_eval_report_exposes_credibility_metadata() -> None:
    report = OfflineEvaluator(settings=_EVAL_OFFLINE).run()
    assert report.notes, "报告必须说明指标的可信度前提"
    assert report.baseline_latency_ms > 0, "基线延迟必须是真实测量值"


# ============================================================
# 回归测试：语义缓存
# ============================================================


def test_l2_recalls_paraphrased_question() -> None:
    """同义改写应命中语义缓存（哈希向量下该用例必然失败）。"""
    if not _semantic_backend_enabled():
        return
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("退款多久能到账", write_back=False)
    assert response.route == QueryRoute.CACHE, (
        f"语义缓存未召回同义改写，route={response.route.value}"
    )


def test_out_of_domain_question_not_cached_reuse() -> None:
    """域外问题不应被语义缓存复用。"""
    if not _semantic_backend_enabled():
        return
    pipeline = build_pipeline(settings=_REDIS_OFF)
    response = pipeline.ask("今天天气怎么样", write_back=False)
    assert response.route != QueryRoute.CACHE


# ============================================================
# LLM 集成（真实调用 DeepSeek，需要 DEEPSEEK_API_KEY）
# ============================================================


def test_llm_enabled_pipeline_routes_to_real_llm() -> None:
    """接入 LLM 后 Validator 应启用 LLM 裁决，而不是走规则降级。"""
    if _SKIP_LLM_TESTS:
        return
    pipeline = build_pipeline(settings=Settings(use_redis=False))
    if not pipeline.validator.use_llm:
        return  # 未配置 key，跳过
    assert pipeline.llm_tracker is not None


def test_llm_token_usage_is_tracked() -> None:
    """真实 token 用量必须被统计到，成本指标才有意义。"""
    if _SKIP_LLM_TESTS:
        return
    pipeline = build_pipeline(settings=Settings(use_redis=False))
    if not pipeline.validator.use_llm:
        return
    pipeline.ask("保修需要提交什么材料？", write_back=False)
    usage = pipeline.llm_tracker.usage
    assert usage.calls > 0, "未统计到任何 LLM 调用"
    assert usage.total_tokens > 0, "未统计到 token 用量"


def test_llm_answer_is_not_empty() -> None:
    """推理模型 token 配额不足时会返回空 content，必须有降级兜底。"""
    if _SKIP_LLM_TESTS:
        return
    pipeline = build_pipeline(settings=Settings(use_redis=False))
    if not pipeline.validator.use_llm:
        return
    response = pipeline.ask("人为损坏可以免费维修吗？", write_back=False)
    assert response.answer.strip(), "LLM 返回了空回答"
