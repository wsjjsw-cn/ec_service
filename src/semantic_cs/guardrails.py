from __future__ import annotations

import re
from dataclasses import dataclass

from semantic_cs.models import Product


@dataclass(frozen=True)
class GuardrailResult:
    is_dynamic: bool
    reason: str | None = None
    product: Product | None = None


class DynamicQueryGuardrail:
    def __init__(self, products: list[Product]) -> None:
        self.products = products
        # 命中这些类别时，即便没有匹配到具体商品也应拦截（否则会漏掉
        # 「预售商品什么时候发货」这类没有商品名但明显动态的问句）
        self.always_dynamic = {"stock", "price", "time", "order_specific", "presale"}
        self.dynamic_patterns = [
            ("stock", re.compile(r"库存|有货|还有几件|现货|仓库")),
            ("price", re.compile(r"价格|多少钱|卖多少|今天.*价|现在.*价")),
            ("time", re.compile(r"今天|现在|当前|实时|此刻")),
            ("order_specific", re.compile(r"订单号|物流单号|我的订单|到哪了|地址改")),
            ("presale", re.compile(r"预售|预定.*发货|预购.*发货|众筹")),
        ]

    def inspect(self, question: str) -> GuardrailResult:
        matched_product = self._match_product(question)
        for reason, pattern in self.dynamic_patterns:
            if pattern.search(question):
                if matched_product or reason in self.always_dynamic:
                    return GuardrailResult(True, reason=reason, product=matched_product)
        return GuardrailResult(False)

    def answer_dynamic(self, question: str, result: GuardrailResult) -> str:
        if result.product and result.reason == "stock":
            product = result.product
            status = "有货" if product.stock > 0 else "暂时无货"
            return (
                f"{product.name} 当前{status}，库存为 {product.stock} 件。"
                f"库存数据更新时间：{product.updated_at}，下单前建议以商品页实时库存为准。"
            )
        if result.product and result.reason == "price":
            product = result.product
            return (
                f"{product.name} 当前参考价为 {product.price:.0f} 元。"
                f"价格数据更新时间：{product.updated_at}，优惠、券后价和活动价请以结算页为准。"
            )
        if result.reason == "presale":
            return (
                "预售商品的发货时间以对应批次为准，不同批次的发货日期可能不同，"
                "不能复用静态答案。请在商品详情页查看该批次的预计发货时间，"
                "或在订单详情页查看发货倒计时。"
            )
        if result.reason == "time":
            return "这个问题涉及实时履约或预售时间，不能复用静态 FAQ。请提供订单号或查看商品页/订单页展示的最新时间。"
        if result.reason == "order_specific":
            return "这个问题需要读取你的订单实时状态，不能使用缓存答案。请在订单详情页查看物流，或提供订单号给人工客服核验。"
        return "该问题包含实时信息，已跳过静态缓存，请以订单页或商品页最新信息为准。"

    def _match_product(self, question: str) -> Product | None:
        normalized_question = question.lower().replace(" ", "")
        for product in self.products:
            model_tokens = "".join(re.findall(r"[a-zA-Z0-9]+", product.name)).lower()
            candidates = {
                product.sku.lower(),
                product.name.lower().replace(" ", ""),
                model_tokens,
            }
            if any(candidate and candidate in normalized_question for candidate in candidates):
                return product
        return None


