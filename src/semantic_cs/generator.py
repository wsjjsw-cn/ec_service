"""
答案生成器 - 基于 LangChain

职责：
1. 使用 LLM 生成答案
2. 答案补全（结合缓存和新检索）
3. 会话记忆管理
"""
from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from semantic_cs.langchain_setup import (
    QA_PROMPT,
    COMPLETE_PROMPT,
    IDENTITY_PROMPT,
    CHITCHAT_PROMPT,
    OOD_PROMPT,
    extract_response_text,
    format_docs_for_prompt,
)
from semantic_cs.models import RetrievalHit


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        use_llm: bool = False,
        max_memory_messages: int = 10,
    ) -> None:
        self.llm = llm
        self.use_llm = use_llm and llm is not None
        self.max_memory_messages = max_memory_messages
        self._session_memory: dict[str, list[BaseMessage]] = {}

    # ============================================================
    # 主生成方法
    # ============================================================
    
    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        session_id: str = "default",
        use_memory: bool = True,
    ) -> str:
        """生成答案"""
        if self.use_llm and self.llm:
            context = self._build_context(hits)
            answer = self._generate_with_llm(question, context, session_id, use_memory)
            # 推理模型在 max_tokens 被推理占满时会返回空 content，此时必须降级，
            # 否则用户会拿到一个空回答
            if not (answer or "").strip():
                print("[Generator] LLM returned empty content, falling back to rules")
                answer = self._rule_based_generate(question, hits)
            self._update_memory(session_id, question, answer)
            return answer
        answer = self._rule_based_generate(question, hits)
        return answer

    def complete_from_cache(
        self,
        question: str,
        cached_answer: str,
        hits: list[RetrievalHit],
        session_id: str = "default",
        use_memory: bool = True,
    ) -> str:
        """补全缓存答案"""
        if self.use_llm and self.llm:
            additional_context = self._build_context(hits)
            answer = self._complete_with_llm(question, cached_answer, additional_context, session_id)
            if not (answer or "").strip():
                answer = f"{cached_answer}\n\n补充说明：{self._rule_based_generate(question, hits)}"
            self._update_memory(session_id, question, answer)
            return answer
        supplement = self._rule_based_generate(question, hits)
        return f"{cached_answer}\n\n补充说明：{supplement}"

    # ============================================================
    # LLM 生成
    # ============================================================
    
    def _generate_with_llm(
        self,
        question: str,
        context: str,
        session_id: str,
        use_memory: bool,
    ) -> str:
        """使用 LLM 生成答案"""
        try:
            if use_memory and session_id in self._session_memory:
                # 带记忆的生成
                messages = list(self._session_memory[session_id])
                messages.append(HumanMessage(content=f"知识库内容：\n{context}\n\n问题：{question}"))
                response = self.llm.invoke(messages)
            else:
                # 不带记忆的生成
                prompt = QA_PROMPT.invoke({
                    "question": question,
                    "context": context,
                })
                response = self.llm.invoke(prompt)
            
            return extract_response_text(response)
        except Exception as e:
            print(f"[Generator] LLM generate error: {e}")
            return self._rule_based_generate(question, [])

    def _complete_with_llm(
        self,
        question: str,
        cached_answer: str,
        additional_context: str,
        session_id: str,
    ) -> str:
        """使用 LLM 补全答案"""
        try:
            prompt = COMPLETE_PROMPT.invoke({
                "existing_answer": cached_answer,
                "additional_context": additional_context,
                "question": question,
            })
            response = self.llm.invoke(prompt)
            return extract_response_text(response)
        except Exception as e:
            print(f"[Generator] LLM complete error: {e}")
            supplement = self._rule_based_generate(question, [])
            return f"{cached_answer}\n\n补充说明：{supplement}"

    # ============================================================
    # 特殊响应生成
    # ============================================================
    
    def generate_identity_response(self, question: str) -> str:
        """生成身份介绍"""
        if self.use_llm and self.llm:
            try:
                prompt = IDENTITY_PROMPT.invoke({"question": question})
                response = self.llm.invoke(prompt)
                return extract_response_text(response)
            except Exception as e:
                print(f"[Generator] Identity error: {e}")
        return self.DEFAULT_IDENTITY_RESPONSE

    def generate_chitchat_response(self, question: str) -> str:
        """生成闲聊响应"""
        if self.use_llm and self.llm:
            try:
                prompt = CHITCHAT_PROMPT.invoke({"question": question})
                response = self.llm.invoke(prompt)
                return extract_response_text(response)
            except Exception as e:
                print(f"[Generator] Chitchat error: {e}")
        return self.DEFAULT_CHITCHAT_RESPONSE

    def generate_out_of_domain_response(self, question: str = "") -> str:
        """生成超出范围响应"""
        if self.use_llm and self.llm and question:
            try:
                prompt = OOD_PROMPT.invoke({"question": question})
                response = self.llm.invoke(prompt)
                return extract_response_text(response)
            except Exception as e:
                print(f"[Generator] OOD error: {e}")
        return self.DEFAULT_OOD_RESPONSE

    # ============================================================
    # 会话记忆
    # ============================================================
    
    def _update_memory(self, session_id: str, question: str, answer: str) -> None:
        """更新会话记忆"""
        if session_id not in self._session_memory:
            self._session_memory[session_id] = []
        
        memory = self._session_memory[session_id]
        memory.append(HumanMessage(content=question))
        memory.append(AIMessage(content=answer))
        
        # 保持记忆长度
        if len(memory) > self.max_memory_messages * 2:
            memory = memory[-self.max_memory_messages * 2:]
        self._session_memory[session_id] = memory

    def get_memory(self, session_id: str) -> list[BaseMessage]:
        """获取会话记忆"""
        return self._session_memory.get(session_id, [])

    def clear_memory(self, session_id: str) -> None:
        """清除会话记忆"""
        if session_id in self._session_memory:
            del self._session_memory[session_id]

    def clear_all_memory(self) -> None:
        """清除所有会话记忆"""
        self._session_memory.clear()

    # ============================================================
    # 工具方法
    # ============================================================
    
    def _build_context(self, hits: list[RetrievalHit]) -> str:
        docs = [hit.doc for hit in hits]
        return format_docs_for_prompt(docs)

    def _rule_based_generate(self, question: str, hits: list[RetrievalHit]) -> str:
        """规则生成（降级方案）"""
        if "专票" in question or "增值税" in question:
            return "增值税专用发票需要填写纳税人识别号、注册地址、电话、开户行和账号。订单完成后可在订单详情页申请。"
        if "微信" in question:
            return "微信支付退款在售后审核通过后通常24小时内原路退回，具体到账时间以微信支付处理结果为准。"
        if "秒杀" in question:
            return "秒杀、限时抢购、赠品变化、优惠券差异等活动价格通常不纳入价保范围。"
        if "质量" in question:
            return "质量问题退货不受七天无理由规则限制，但需要提交照片、视频等凭证并通过售后审核。"
        if "过期" in question and "优惠券" in question:
            return "退款后优惠券是否返还取决于券状态；若优惠券已过有效期，通常不会返还。"
        if "偏远" in question:
            return "偏远地区物流一般需要5到7天，大促、天气或交通管制可能导致进一步延迟。"
        if "人为" in question or "维修" in question:
            return "保修或维修申请需要订单号、故障描述以及照片或视频凭证；人为损坏、进水、私自拆修通常不在免费保修范围内。"
        
        if hits:
            context = self._build_context(hits)
            first_sentence = context.split("。", 1)[0].replace("\n", " ")
            return f"{first_sentence}。"
        return "抱歉，我暂时无法回答这个问题。"

    @staticmethod
    def is_low_confidence(hits: list[RetrievalHit], threshold: float = 0.15) -> bool:
        if not hits:
            return True
        return hits[0].score < threshold

    # ============================================================
    # 默认响应（降级）
    # ============================================================
    
    DEFAULT_IDENTITY_RESPONSE = (
        "您好，我是智能电商客服助手小智。我可以帮您解答商品咨询、"
        "售后政策、订单查询、物流追踪等问题。请问有什么可以帮您？"
    )

    DEFAULT_CHITCHAT_RESPONSE = (
        "您好！作为电商客服，我主要负责解答购物相关的问题，比如商品、订单、售后、物流等。"
        "请问您有什么购物问题需要帮助？"
    )

    DEFAULT_OOD_RESPONSE = (
        "抱歉，我是电商客服小智，主要解答商品、订单、售后、物流等购物相关问题。"
        "您的问题可能不在我的服务范围内，建议您咨询相关客服或查看帮助中心。"
    )
