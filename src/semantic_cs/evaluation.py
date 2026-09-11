"""离线评测：真实测量缓存中间件的收益。

与旧实现的本质区别：

* 旧版延迟来自硬编码常量 ``_synthetic_latency``（20/45/80/180/290ms），
  ``baseline_latency=250.0`` 与 ``baseline_llm_calls=51`` 同样是拍脑袋的常量，
  ``started = time.perf_counter()`` 定义后从未被使用 —— 所谓「成本节省 32%、
  吞吐提升 36%」只是常量算术的结果。
* 新版：延迟用真实墙钟时间；基线用同一 pipeline 关闭缓存（``bypass_cache=True``）
  真实跑一遍；成本按实测（或按链路估算）的 LLM 调用次数折算。

评测使用独立的内存 pipeline，并在每轮开始前把缓存重置到「仅预加载 FAQ」状态，
因此 ``/metrics/eval`` 是幂等的，且不会污染线上会话缓存。
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, replace

from semantic_cs.bootstrap import build_pipeline, preload_hot_faqs
from semantic_cs.config import Settings, get_settings
from semantic_cs.data_loader import load_eval_requests
from semantic_cs.llm_usage import CostModel, TokenUsage, TokenUsageTracker
from semantic_cs.models import QueryRoute
from semantic_cs.pipeline import CustomerServicePipeline

# 未接入真实 LLM 时，按链路估算各路径的调用次数，避免整条链路都是 0 导致节省率失真
_ROUTE_LLM_CALL_ESTIMATE = {
    QueryRoute.DYNAMIC: 0,
    QueryRoute.IDENTITY: 0,
    QueryRoute.CHITCHAT: 0,
    QueryRoute.OOD: 1,
}


@dataclass(frozen=True)
class EvalReport:
    total: int
    dynamic_intercepts: int
    cache_coverage_rate: float
    direct_cache_reuse_rate: float
    avg_latency_ms: float
    baseline_latency_ms: float
    latency_reduction_rate: float
    throughput_multiplier: float
    avg_llm_saved: float
    total_llm_saved: int
    cost_saving_rate: float
    # ---- 新增：可信度元数据 ----
    llm_enabled: bool
    embedding_backend: str
    llm_calls_baseline: float
    llm_calls_actual: float
    # 真实 token 计量（接入 LLM 时有效）
    baseline_input_tokens: int
    baseline_output_tokens: int
    actual_input_tokens: int
    actual_output_tokens: int
    baseline_cost_usd: float
    actual_cost_usd: float
    # 稳态轮：缓存已积累上一轮回写的内容。命中率高是「重复问同一句」的必然结果，
    # 单独列出，不作为主指标，避免高估系统能力。
    warm_cache_coverage_rate: float
    warm_direct_cache_reuse_rate: float
    notes: list[str] = field(default_factory=list)


@dataclass
class _RoundResult:
    latencies: list[float]
    cache_hits: int
    direct_reuse: int
    dynamic_intercepts: int
    llm_calls: float
    usage: TokenUsage | None = None


class OfflineEvaluator:
    def __init__(
        self,
        pipeline: CustomerServicePipeline | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._pipeline = pipeline
        self._eval_pipeline: CustomerServicePipeline | None = None

    # ============================================================
    # 评测专用 pipeline：独立、可重置，保证幂等
    # ============================================================

    def _owned_pipeline(self) -> CustomerServicePipeline:
        if self._pipeline is not None:
            return self._pipeline
        if self._eval_pipeline is None:
            # 必须基于 get_settings()（读取 .env 的真实 LLM_MODEL/API_KEY 等），
            # 仅把 use_redis 覆盖为 False 以使用隔离的内存 store。
            # 旧实现直接传 Settings(use_redis=False)，那是默认的 MiniMax-M2.5 配置，
            # 会导致切到 deepseek-flash 后仍用 MiniMax 模型初始化 LLM（已修复）。
            eval_settings = get_settings()
            self._eval_pipeline = build_pipeline(
                preload=True, settings=replace(eval_settings, use_redis=False)
            )
        self._reset_cache(self._eval_pipeline)
        return self._eval_pipeline

    def _reset_cache(self, pipeline: CustomerServicePipeline) -> None:
        """把缓存恢复到「仅预加载高频 FAQ」的初始状态。

        同时清空 embedding 编码缓存：否则先跑的轮次会给后跑的轮次预热，
        让后面的轮次延迟虚低（真实语义模型单次编码可达数十毫秒）。
        """
        pipeline.store.clear()
        pipeline.embedding_model.clear_cache()
        preload_hot_faqs(pipeline.store, pipeline.embedding_model)
        pipeline.clear_all_memory()

    # ============================================================
    # 主流程
    # ============================================================

    def run(self) -> EvalReport:
        requests = load_eval_requests()
        pipeline = self._owned_pipeline()
        questions = [item["question"] for item in requests]
        llm_enabled = bool(pipeline.validator.use_llm)
        notes: list[str] = []

        # 1. 基线：关闭缓存中间件，每条都走完整 RAG
        baseline_latencies: list[float] = []
        baseline_calls = 0.0
        pipeline.embedding_model.clear_cache()
        if pipeline.llm_tracker:
            pipeline.llm_tracker.reset()
        for index, question in enumerate(questions):
            started = time.perf_counter()
            response = pipeline.ask(
                question,
                session_id=f"eval-baseline-{index}",
                write_back=False,
                use_memory=False,
                bypass_cache=True,
            )
            baseline_latencies.append((time.perf_counter() - started) * 1000)
            baseline_calls += self._count_llm_calls(response, llm_enabled)

        baseline_usage = (
            pipeline.llm_tracker.usage if pipeline.llm_tracker else None
        )

        # 2. 冷启动一轮（会写回缓存）
        cold = self._run_round(pipeline, questions, session_prefix="eval-cold", reset_usage=True)

        # 3. 稳态一轮（缓存已积累上一轮回写的内容）
        warm = self._run_round(
            pipeline, questions, session_prefix="eval-warm", reset_usage=True
        )

        total = len(questions)
        baseline_latency = statistics.mean(baseline_latencies) if baseline_latencies else 0.0
        # 主指标取冷启动轮：用户第一次提问时的真实命中能力。
        # 稳态轮因为缓存里已有完全相同的问句，命中率高但参考价值低。
        avg_latency = statistics.mean(cold.latencies) if cold.latencies else 0.0
        latency_reduction = (
            max(0.0, (baseline_latency - avg_latency) / baseline_latency)
            if baseline_latency
            else 0.0
        )
        throughput = baseline_latency / max(1.0, avg_latency)

        actual_calls = cold.llm_calls
        saved = max(0.0, baseline_calls - actual_calls)

        # 成本：接入 LLM 时按真实 token 计量，否则退化为调用次数估算
        cost_model = CostModel(
            input_price_per_mtok=self.settings.llm_input_cost_per_mtok,
            output_price_per_mtok=self.settings.llm_output_cost_per_mtok,
            cache_lookup_cost=self.settings.cache_lookup_cost,
        )
        baseline_usage_obj = baseline_usage or TokenUsage()
        actual_usage_obj = cold.usage or TokenUsage()
        if llm_enabled and (baseline_usage_obj.calls or actual_usage_obj.calls):
            baseline_cost = cost_model.total(baseline_usage_obj, requests=total)
            actual_cost = cost_model.total(actual_usage_obj, requests=total)
        else:
            baseline_cost = baseline_calls * self.settings.llm_call_cost
            actual_cost = (
                actual_calls * self.settings.llm_call_cost
                + total * self.settings.cache_lookup_cost
            )
        cost_saving = (baseline_cost - actual_cost) / baseline_cost if baseline_cost else 0.0

        dynamic_intercepts = cold.dynamic_intercepts
        denominator = max(1, total - dynamic_intercepts)

        if not llm_enabled:
            notes.append(
                "未配置 LLM_API_KEY：系统运行在规则降级模式，延迟与成本为本地链路测量值，"
                "不代表接入 LLM 后的真实收益；LLM 调用数按链路估算。"
            )
        else:
            if baseline_usage_obj.total_tokens == 0 and actual_usage_obj.total_tokens == 0:
                notes.append(
                    "LLM 调用疑似全部失败（鉴权/网络错误，usage 为 0）：token 与成本数字不可信，"
                    "本结果实为规则降级模式，不能代表接入 LLM 后的真实收益；请检查 LLM_API_KEY 是否有效。"
                )
            else:
                notes.append(
                    f"成本按真实 token 计量：基线 {baseline_usage_obj.total_tokens} tokens "
                    f"vs 缓存后 {actual_usage_obj.total_tokens} tokens"
                    f"（其中推理 token {actual_usage_obj.reasoning_tokens}）。"
                )
        notes.append(
            "覆盖率/复用率取冷启动轮（用户首次提问的真实命中能力）；稳态轮为 "
            f"{warm.cache_hits / denominator:.2%} / {warm.direct_reuse / denominator:.2%}，"
            "该值偏高是因为缓存里已存在完全相同的问句，不代表泛化能力。"
        )
        notes.append(f"embedding 后端：{getattr(pipeline.embedding_model, 'backend_name', 'unknown')}")

        return EvalReport(
            total=total,
            dynamic_intercepts=dynamic_intercepts,
            cache_coverage_rate=cold.cache_hits / denominator,
            direct_cache_reuse_rate=cold.direct_reuse / denominator,
            avg_latency_ms=avg_latency,
            baseline_latency_ms=baseline_latency,
            latency_reduction_rate=latency_reduction,
            throughput_multiplier=throughput,
            avg_llm_saved=saved / max(1, total),
            total_llm_saved=int(saved),
            cost_saving_rate=cost_saving,
            llm_enabled=llm_enabled,
            embedding_backend=getattr(pipeline.embedding_model, "backend_name", "unknown"),
            llm_calls_baseline=baseline_calls,
            llm_calls_actual=actual_calls,
            baseline_input_tokens=baseline_usage_obj.input_tokens,
            baseline_output_tokens=baseline_usage_obj.output_tokens,
            actual_input_tokens=actual_usage_obj.input_tokens,
            actual_output_tokens=actual_usage_obj.output_tokens,
            baseline_cost_usd=baseline_cost,
            actual_cost_usd=actual_cost,
            warm_cache_coverage_rate=warm.cache_hits / denominator,
            warm_direct_cache_reuse_rate=warm.direct_reuse / denominator,
            notes=notes,
        )

    # ============================================================
    # 辅助
    # ============================================================

    def _run_round(
        self,
        pipeline: CustomerServicePipeline,
        questions: list[str],
        session_prefix: str,
        write_back: bool = True,
        reset_usage: bool = False,
    ) -> _RoundResult:
        llm_enabled = bool(pipeline.validator.use_llm)
        latencies: list[float] = []
        cache_hits = 0
        direct_reuse = 0
        dynamic_intercepts = 0
        llm_calls = 0.0

        pipeline.embedding_model.clear_cache()  # 逐轮冷启动，保证延迟可比
        if reset_usage and pipeline.llm_tracker:
            pipeline.llm_tracker.reset()

        for index, question in enumerate(questions):
            started = time.perf_counter()
            response = pipeline.ask(
                question,
                session_id=f"{session_prefix}-{index}",
                write_back=write_back,
                use_memory=False,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            llm_calls += self._count_llm_calls(response, llm_enabled)
            if response.route == QueryRoute.DYNAMIC:
                dynamic_intercepts += 1
            elif response.route == QueryRoute.CACHE:
                cache_hits += 1
                if response.cache_level in {"L1", "L2"}:
                    direct_reuse += 1

        return _RoundResult(
            latencies=latencies,
            cache_hits=cache_hits,
            direct_reuse=direct_reuse,
            dynamic_intercepts=dynamic_intercepts,
            llm_calls=llm_calls,
            usage=pipeline.llm_tracker.usage if pipeline.llm_tracker else None,
        )

    def _count_llm_calls(self, response, llm_enabled: bool) -> float:
        """统计 LLM 调用次数。

        接入 LLM 时用链路真实上报值；未接入时按路径估算，
        否则整条链路都是 0，节省率会失真成 100%。
        """
        if llm_enabled:
            return float(response.llm_calls)
        if response.route in _ROUTE_LLM_CALL_ESTIMATE:
            return float(_ROUTE_LLM_CALL_ESTIMATE[response.route])
        if response.route == QueryRoute.CACHE:
            if response.cache_level == "L1":
                return 0.0
            if response.cache_level == "L2":
                return 1.0  # Validator 裁决
            return 2.0  # L2+RAG：Validator + 补全生成
        return float(self.settings.rag_llm_calls_per_query)
