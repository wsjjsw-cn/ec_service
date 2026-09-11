"""
LLM Validator - 使用 LLM 判断缓存答案质量

职责：
1. 判断缓存答案是否可以直接复用（REUSE）
2. 判断是否需要补充信息（COMPLETE）
3. 判断是否需要完整 RAG（RAG）
4. 判断问题是否在服务范围内（IN_SCOPE / OUT_OF_SCOPE）
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_core.language_models import BaseChatModel

from semantic_cs.langchain_setup import (
    VALIDATE_COMPLETENESS_PROMPT,
    VALIDATE_OOD_PROMPT,
    extract_response_text,
)
from semantic_cs.models import CacheDecision, SemanticCandidate, ValidationResult


class LLMValidator:
    """基于 LLM 的答案验证器"""
    
    def __init__(
        self,
        reuse_threshold: float = 0.80,
        complete_threshold: float = 0.65,
        llm: Optional[BaseChatModel] = None,
        use_llm: bool = False,
        reuse_coverage_threshold: float = 0.6,
    ) -> None:
        self.reuse_threshold = reuse_threshold
        self.complete_threshold = complete_threshold
        self.llm = llm
        self.use_llm = use_llm and llm is not None
        # 新问句被缓存问句覆盖的词占比，低于该值说明引入了新的信息诉求，不宜直接复用
        self.reuse_coverage_threshold = reuse_coverage_threshold

    def validate(self, question: str, candidates: list[SemanticCandidate]) -> ValidationResult:
        """验证候选答案质量"""
        if not candidates:
            return ValidationResult(CacheDecision.RAG, None, [], 0.0)
        
        best = max(candidates, key=lambda item: item.score)
        
        # 如果启用 LLM，使用 LLM 判断
        if self.use_llm and self.llm:
            return self._validate_with_llm(question, best)
        
        # 规则降级
        return self._validate_with_rules(question, best)
    
    def _validate_with_llm(self, question: str, best: SemanticCandidate) -> ValidationResult:
        """使用 LLM 验证"""
        try:
            prompt = VALIDATE_COMPLETENESS_PROMPT.invoke({
                "original_question": best.entry.question,
                "original_answer": best.entry.answer,
                "new_question": question,
            })
            response = self.llm.invoke(prompt)
            result_text = extract_response_text(response).upper()
            
            # 根据 LLM 判断决定
            # 注意：candidate 必须是 SemanticCandidate 本身（pipeline 按 .entry.answer 解包），
            # 传 best.entry 会导致 AttributeError —— 该分支在旧哈希向量下从未被触发，故一直没暴露
            if "REUSE" in result_text:
                return ValidationResult(
                    CacheDecision.REUSE, best, [], best.score
                )
            elif "COMPLETE" in result_text:
                missing = self._extract_missing_info(result_text)
                return ValidationResult(
                    CacheDecision.COMPLETE, best, missing, best.score
                )
            else:
                return ValidationResult(CacheDecision.RAG, None, [], best.score)
        except Exception as e:
            print(f"[Validator] LLM validation failed: {e}, falling back to rules")
            return self._validate_with_rules(question, best)
    
    def _validate_with_rules(self, question: str, best: SemanticCandidate) -> ValidationResult:
        """规则降级验证。

        注意：不能简单用「差集非空」判定缺失 —— 同义改写几乎总会引入新词，
        那样会导致缓存永远判 COMPLETE 而走不到直接复用。这里改用覆盖率：
        新问句被缓存问句覆盖的词占比达标才允许直接复用。
        """
        missing_terms = self._missing_intent_terms(question, best.entry.question)
        coverage = self._term_coverage(question, best.entry.question)

        if (
            best.score >= self.reuse_threshold
            and coverage >= self.reuse_coverage_threshold
        ):
            return ValidationResult(
                CacheDecision.REUSE, best, missing_terms, best.score
            )
        if best.score >= self.complete_threshold:
            return ValidationResult(
                CacheDecision.COMPLETE, best, missing_terms, best.score
            )
        return ValidationResult(CacheDecision.RAG, None, missing_terms, best.score)

    @staticmethod
    def _term_coverage(question: str, cached_question: str) -> float:
        """新问句被缓存问句覆盖的比例（字级）。

        用字级而不是连续词：中文同义改写会把「需要满足什么条件」改成「有什么要求」，
        连续词几乎不重叠，会导致改写句永远判不进直接复用。
        """
        question_chars = set(re.findall(r"[\u4e00-\u9fa5]", question))
        if not question_chars:
            return 1.0
        cached_chars = set(re.findall(r"[\u4e00-\u9fa5]", cached_question))
        return len(question_chars & cached_chars) / len(question_chars)
    
    def is_out_of_scope(self, question: str) -> bool:
        """使用 LLM 判断是否超出服务范围"""
        if self.use_llm and self.llm:
            try:
                prompt = VALIDATE_OOD_PROMPT.invoke({"question": question})
                response = self.llm.invoke(prompt)
                result_text = extract_response_text(response).upper()
                return "OUT_OF_SCOPE" in result_text
            except Exception as e:
                print(f"[Validator] OOD check failed: {e}, using rule-based")
                return self._is_out_of_scope_rules(question)
        return self._is_out_of_scope_rules(question)
    
    def _is_out_of_scope_rules(self, question: str) -> bool:
        """规则降级 OOD 判断"""
        # 包含业务关键词的问题认为在范围内
        business_keywords = [
            '退', '换', '修', '保', '发票', '优惠', '券', '价', '物流',
            '快递', '订单', '支付', '款', '商品', '货', '售后', '客服',
            '会员', '积分', '地址', '配送', '签收', '购买', '下单',
            '尺码', '颜色', '规格', '型号', '质量', '正品', '行货',
            '保修', '维修', '清洁', '保养', '使用', '安装', '设置',
            '退货', '换货', '退款', '价保', '保价', '包邮', '发货',
            '到货', '库存', '有货', '现货'
        ]
        
        for keyword in business_keywords:
            if keyword in question:
                return False
        return True
    
    @staticmethod
    def _missing_intent_terms(question: str, cached_question: str) -> list[str]:
        """提取缺失的意图词"""
        question_tokens = set(re.findall(r'[\u4e00-\u9fa5]+', question))
        cached_tokens = set(re.findall(r'[\u4e00-\u9fa5]+', cached_question))
        return list(question_tokens - cached_tokens)
    
    @staticmethod
    def _extract_missing_info(response_text: str) -> list[str]:
        """从 LLM 响应中提取缺失信息"""
        missing = []
        if "缺少" in response_text or "缺失" in response_text:
            # 简单提取引号内的内容
            matches = re.findall(r'["\u201c](.+?)["\u201d]', response_text)
            missing.extend(matches)
        return missing
