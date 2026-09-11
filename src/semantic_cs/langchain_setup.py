"""
LangChain + LangGraph 核心模块

- LLM 初始化
- Prompt 模板管理
- 类型定义
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 加载 .env 文件（项目根目录优先，其次是包目录）
# override=True：.env 是本项目的配置真源，必须覆盖运行环境里可能残留的旧 LLM_* 变量，
# 否则切模型/换 key 后改动不会生效（曾出现沙箱残留的旧 LLM_API_KEY 顶掉 .env 的问题）。
_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
_PACKAGE_ENV = Path(__file__).parent / ".env"
for _env_path in (_ROOT_ENV, _PACKAGE_ENV):
    if _env_path.exists():
        load_dotenv(_env_path, override=True)


# ============================================================
# LLM 工厂函数
# ============================================================

def create_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    callbacks: Optional[list] = None,
    provider: Optional[str] = None,
) -> Optional[BaseChatModel]:
    """创建 LLM 实例。

    支持两类后端：
    - DeepSeek 官方：``ChatDeepSeek``（provider=deepseek）
    - OpenAI 兼容网关（SiliconFlow 等）：``ChatOpenAI``（provider=openai）

    provider=auto 时按 base_url 自动判断。callbacks 用于挂载 token 用量追踪，
    见 semantic_cs.llm_usage。
    """
    api_key = (
        api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    base_url = _normalize_base_url(
        base_url
        or os.getenv("LLM_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    model = (
        model
        or os.getenv("LLM_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-v4-flash"
    )
    provider = (provider or os.getenv("LLM_PROVIDER", "auto")).lower()
    if provider == "auto":
        provider = "deepseek" if "deepseek" in base_url.lower() else "openai"

    if not api_key:
        print("[LLM] Warning: no API key configured (LLM_API_KEY / DEEPSEEK_API_KEY)")
        return None

    if provider == "deepseek":
        try:
            from langchain_deepseek import ChatDeepSeek

            llm = ChatDeepSeek(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )
            print(f"[LLM] Initialized: {model} (deepseek)")
            return llm
        except ImportError:
            print("[LLM] langchain_deepseek not available, falling back to ChatOpenAI")
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return None

    return _create_openai_llm(api_key, base_url, model, temperature, max_tokens, callbacks)


def _create_openai_llm(
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    callbacks: Optional[list] = None,
) -> Optional[BaseChatModel]:
    """OpenAI 兼容后端（SiliconFlow、vLLM、OneAPI 等）。"""
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
        print(f"[LLM] Initialized: {model} (openai-compatible @ {base_url})")
        return llm
    except ImportError:
        print("[LLM] No compatible LLM package found (pip install langchain-openai)")
        return None
    except Exception as e:
        print(f"[LLM] Fallback error: {e}")
        return None


def _normalize_base_url(base_url: str) -> str:
    """去掉结尾斜杠。

    旧实现在兜底分支里又拼了一次 "/v1"，当 base_url 已经是 ".../v1" 时会变成
    ".../v1/v1" 导致 404，这里统一只保留调用方给定的路径。
    """
    return base_url.rstrip("/")


# ============================================================
# Prompt 模板
# ============================================================

SYSTEM_PROMPT_CUSTOMER_SERVICE = """你是「智能电商客服」，名为小智。

## 角色定位
- 你是一个专业、友好、高效的电商客服助手
- 服务范围：商品咨询、订单管理、售后服务、物流查询、促销活动

## 回答规范
1. **基于知识**：优先使用提供的知识库内容，确保回答准确
2. **诚实透明**：如果知识库没有相关内容，明确告知用户，不要编造信息
3. **简洁明了**：用通俗易懂的语言，避免专业术语
4. **友好礼貌**：使用亲切的语气，让用户感到温暖

## 特殊处理
- 对于问候语（你好、您好）：友好回应并询问需求
- 对于感谢语（谢谢、感谢）：礼貌回应
- 对于告别语（再见、拜拜）：友好道别
- 对于超出服务范围的问题：礼貌拒绝并引导到正确渠道"""


# 主问答 Prompt
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_CUSTOMER_SERVICE),
    ("human", """## 知识库内容
{context}

## 用户问题
{question}

## 回答要求
请根据知识库内容，给出准确、简洁、友好的回答。"""),
])


# 答案补全 Prompt
COMPLETE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_CUSTOMER_SERVICE),
    ("human", """## 已有答案
{existing_answer}

## 补充知识
{additional_context}

## 用户问题
{question}

## 任务
请综合已有答案和补充知识，生成一个更完整、更准确的回答。如果已有答案已经很完整，可以在其基础上添加补充内容。"""),
])


# Validator: 判断缓存答案是否完整
VALIDATE_COMPLETENESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的答案评估专家。你需要判断一个缓存的答案是否可以直接复用来回答新的问题。

评估维度：
1. **完整性**：缓存答案是否覆盖了新问题的所有方面？
2. **准确性**：缓存答案是否仍然准确？
3. **相关性**：缓存答案是否与新问题高度相关？

请给出判断结果和理由。"""),
    ("human", """## 原始问题（缓存）
{original_question}

## 原始答案（缓存）
{original_answer}

## 新问题
{new_question}

## 判断
请判断：
- 如果可以直接复用，请回复「REUSE」并给出简要理由
- 如果可以部分复用但需要补充，请回复「COMPLETE」并说明需要补充什么
- 如果不能复用，请回复「RAG」"""),
])


# Validator: 判断是否超出知识库范围
VALIDATE_OOD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个电商客服意图识别专家。你需要判断用户的问题是否属于电商服务范畴。

电商服务范畴包括：
- 商品信息（价格、规格、库存、功能等）
- 订单管理（下单、支付、取消、修改等）
- 售后服务（退货、换货、维修、保修等）
- 物流查询（发货、配送、签收等）
- 促销活动（优惠券、折扣、积分等）

不属于的范畴：
- 闲聊（天气、新闻、八卦等）
- 其他领域的问题（医疗、法律、编程等）
- 需要实时数据的问题（股票、汇率等）"""),
    ("human", """## 用户问题
{question}

## 判断
请判断该问题是否属于电商服务范畴：
- 如果属于，请回复「IN_SCOPE」
- 如果不属于，请回复「OUT_OF_SCOPE」"""),
])


# ReAct: 问题分析
ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个问题分析专家，用一句话简短回答。"),
    ("human", """用户问题：{question}

请分析这个问题需要什么类型的知识来回答？"""),
])


# ReAct: 判断信息是否足够
JUDGE_SUFFICIENCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息判断专家，用简短的话回答。"),
    ("human", """## 检索结果摘要
{context_summary}

## 用户问题
{question}

判断检索结果是否有足够的信息来回答问题？（是/否）"""),
])


# 身份介绍 Prompt
IDENTITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_CUSTOMER_SERVICE),
    ("human", """用户问：「{question}」

请用一段简短的话介绍你自己。"""),
])


# 闲聊回复 Prompt
CHITCHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_CUSTOMER_SERVICE),
    ("human", """用户说：「{question}」

请用一句话友好回应，然后引导用户提出具体的购物问题。"""),
])


# 超出范围回复 Prompt
OOD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是「智能电商客服」，名为小智。你只回答与电商购物相关的问题。

如果用户的问题超出你的服务范围，请：
1. 礼貌地说明你只能回答购物相关的问题
2. 友好地引导用户提出商品、订单、售后等方面的问题
3. 语气要亲切，不要生硬拒绝"""),
    ("human", """用户问：「{question}」

这个问题超出了电商购物的范畴，请用友好、亲切的方式回应，并引导用户提出购物相关的问题。"""),
])


# ============================================================
# 工具函数
# ============================================================

def format_docs_for_prompt(docs: list, max_docs: int = 3) -> str:
    """格式化文档供 Prompt 使用"""
    formatted = []
    for i, doc in enumerate(docs[:max_docs], 1):
        # 兼容不同的文档类型
        if hasattr(doc, 'metadata'):
            title = doc.metadata.get('title', '') or f'文档{i}'
            body = doc.page_content or ''
        elif hasattr(doc, 'title'):
            title = doc.title or f'文档{i}'
            body = doc.body or getattr(doc, 'page_content', '')
        else:
            title = f'文档{i}'
            body = str(doc)
        
        # 截断过长内容
        body = body[:500] if len(body) > 500 else body
        formatted.append(f"【{title}】\n{body}")
    
    return "\n\n".join(formatted) if formatted else "（无相关文档）"


def extract_response_text(response) -> str:
    """从 LLM 响应中提取文本"""
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, str):
        return response
    return str(response)
