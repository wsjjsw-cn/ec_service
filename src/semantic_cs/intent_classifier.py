from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    IDENTITY = "identity"
    CHITCHAT = "chitchat"
    BUSINESS = "business"
    OUT_OF_DOMAIN = "out_of_domain"


@dataclass(frozen=True)
class IntentResult:
    intent: IntentType
    confidence: float
    reason: str = ""


class IntentClassifier:
    _IDENTITY_KEYWORDS = [
        "你是谁", "你是啥", "你是什么", "你是哪位", "你是做什么",
        "介绍自己", "自我介绍", "你的功能", "你的作用",
    ]

    _CHITCHAT_KEYWORDS = [
        "你好", "您好", "哈喽", "hello", "hi", "hey", "嗨",
        "谢谢", "感谢", "多谢", "thanks", "thank you",
        "再见", "拜拜", "bye bye", "bye",
        "在吗", "有空吗", "忙吗",
    ]

    # 注意：关键词表缺词会直接把业务问题误判为 OUT_OF_DOMAIN（例如「增值税专票」
    # 不含「发票」二字、「积分可以提现吗」不含「会员」），这里按真实问法补全。
    _BUSINESS_KEYWORDS = [
        "退", "换", "保", "修", "发票", "价", "款", "货", "单",
        "物流", "到账", "支付", "包邮", "七天", "无理由", "质量",
        "商品", "订单", "售后", "会员", "优惠", "券", "评价",
        "配送", "退货", "换货", "保修", "维修", "价保", "退款",
        "发货", "下单", "改地址", "取消", "超时", "缺货", "库存",
        "正品", "假", "投诉", "赠品", "活动", "客服", "快递",
        "签收", "安装", "配件", "延保", "换新",
        # 补：发票/税务相关
        "专票", "开票", "抬頭", "抬头", "税号", "纳税", "报销",
        # 补：积分/会员权益
        "积分", "提现", "抵扣", "兑换", "等级",
        # 补：售后凭证与场景
        "凭证", "人为", "损坏", "材料", "照片", "视频", "鉴定",
        # 补：物流与促销
        "偏远", "大促", "秒杀", "预售", "分期", "白条", "运费",
    ]

    def classify(self, question: str) -> IntentResult:
        if self._is_identity(question):
            return IntentResult(IntentType.IDENTITY, 1.0, "identity_keyword_match")

        if self._is_chitchat(question):
            return IntentResult(IntentType.CHITCHAT, 1.0, "chitchat_keyword_match")

        if self._is_business(question):
            return IntentResult(IntentType.BUSINESS, 0.9, "business_keyword_match")

        return IntentResult(IntentType.OUT_OF_DOMAIN, 0.8, "no_domain_keywords")

    def _is_identity(self, question: str) -> bool:
        return any(keyword in question for keyword in self._IDENTITY_KEYWORDS)

    def _is_chitchat(self, question: str) -> bool:
        return any(keyword.lower() in question.lower() for keyword in self._CHITCHAT_KEYWORDS)

    def _is_business(self, question: str) -> bool:
        return any(keyword in question for keyword in self._BUSINESS_KEYWORDS)
