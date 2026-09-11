"""LLM token 用量追踪。

成本节省应该按真实 token 计量，而不是「调用次数 × 固定单价」——
推理模型（如 DeepSeek V4 Flash / MiniMax-M2.5）单次调用的输出 token 里绝大部分是
reasoning（实测占比可超 90%），只统计调用次数会严重低估缓存的收益，也会掩盖推理开销。

通过 LangChain 回调挂载到 LLM 上，覆盖 Validator、生成器、ReAct Agent 的全部调用。
"""
from __future__ import annotations

from dataclasses import dataclass

try:  # LangChain 可用时继承官方基类，保证被当作合法 handler 处理
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
except Exception:  # pragma: no cover - 无 LangChain 时退化为普通类
    _BaseCallbackHandler = object


@dataclass
class TokenUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        calls: int = 1,
    ) -> None:
        self.calls += calls
        self.input_tokens += max(0, int(input_tokens or 0))
        self.output_tokens += max(0, int(output_tokens or 0))
        self.reasoning_tokens += max(0, int(reasoning_tokens or 0))

    def cost_usd(self, input_price_per_mtok: float, output_price_per_mtok: float) -> float:
        """按每百万 token 单价计算成本（USD）。"""
        return (
            self.input_tokens / 1_000_000 * input_price_per_mtok
            + self.output_tokens / 1_000_000 * output_price_per_mtok
        )


class TokenUsageTracker(_BaseCallbackHandler):
    """LangChain 回调：累计每次 LLM 调用的 token 用量。

    继承 BaseCallbackHandler 是必要的 —— LangChain 1.x 在部分代码路径上会按
    manager 接口访问回调对象（如 raise_error），纯 duck-typing 的类会报错。
    """

    def __init__(self) -> None:
        super().__init__()
        self.usage = TokenUsage()

    def reset(self) -> None:
        self.usage = TokenUsage()

    # ---- LangChain 回调接口 ----

    def on_llm_end(self, response, **kwargs) -> None:
        self.usage.add(**_extract_usage(response))

    async def on_llm_end_async(self, response, **kwargs) -> None:
        self.usage.add(**_extract_usage(response))


def _extract_usage(response) -> dict:
    """从不同版本的 LangChain 响应里提取 token 用量。

    langchain-core 各版本放置 usage 的位置不一致：可能在 llm_output 的
    token_usage / usage 字典里，也可能挂在 AIMessage.usage_metadata 上。
    """
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    llm_output = getattr(response, "llm_output", None) or {}
    for key in ("token_usage", "usage"):
        raw = llm_output.get(key) if isinstance(llm_output, dict) else None
        if isinstance(raw, dict):
            usage["input_tokens"] = raw.get("prompt_tokens", 0) or 0
            usage["output_tokens"] = raw.get("completion_tokens", 0) or 0
            details = raw.get("completion_tokens_details") or {}
            usage["reasoning_tokens"] = details.get("reasoning_tokens", 0) or 0
            return usage

    try:
        generations = getattr(response, "generations", None) or []
        message = generations[0][0].message
        meta = getattr(message, "usage_metadata", None) or {}
        if meta:
            usage["input_tokens"] = meta.get("input_tokens", 0) or 0
            usage["output_tokens"] = meta.get("output_tokens", 0) or 0
            details = meta.get("output_token_details") or {}
            usage["reasoning_tokens"] = details.get("reasoning", 0) or 0
            return usage
    except Exception:
        pass

    return usage


@dataclass
class CostModel:
    """计费参数（USD / 每百万 token）。默认走 SiliconFlow MiniMax-M2.5 定价：
    输入 $0.3/M、输出 $1.2/M。评测时由 settings 覆盖为实际模型价格。"""

    input_price_per_mtok: float = 0.30
    output_price_per_mtok: float = 1.20
    # 每次请求的缓存查询开销（向量检索 + Redis/内存查找），按 USD 计
    cache_lookup_cost: float = 0.00057

    def total(self, usage: TokenUsage, requests: int = 0) -> float:
        return usage.cost_usd(
            self.input_price_per_mtok, self.output_price_per_mtok
        ) + requests * self.cache_lookup_cost
