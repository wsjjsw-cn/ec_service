"""
基于 LangGraph 的 ReAct Agent 实现

支持：
1. 会话记忆
2. 多步骤推理
3. 工具调用（知识库检索）
"""
from __future__ import annotations

from typing import TypedDict, Annotated, Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel

from semantic_cs.langchain_setup import (
    ANALYZE_PROMPT,
    JUDGE_SUFFICIENCY_PROMPT,
    extract_response_text,
    format_docs_for_prompt,
)
from semantic_cs.models import RetrievalHit


# ============================================================
# State 定义
# ============================================================

class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    context: str
    thoughts: list[str]
    actions: list[str]
    observations: list[str]
    final_answer: str
    llm_calls: int


# ============================================================
# LangGraph ReAct Agent
# ============================================================

class LangGraphReActAgent:
    """基于 LangGraph 的 ReAct Agent"""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        retriever=None,
        top_k: int = 4,
        use_llm: bool = True,
        max_memory_messages: int = 10,
        callbacks: Optional[list] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.use_llm = use_llm and llm is not None
        self.max_memory_messages = max_memory_messages
        self.callbacks = callbacks or []
        
        # 会话记忆
        self._session_memory: dict[str, list[BaseMessage]] = {}
        
        # 创建 Agent
        if self.use_llm and self.retriever:
            self.graph = self._build_with_tools()
        else:
            self.graph = None

    def _build_with_tools(self):
        """构建带工具的 Agent"""
        
        @tool
        def search_knowledge_base(query: str) -> str:
            """从知识库中搜索相关信息。输入搜索关键词，返回找到的相关文档。"""
            hits = self.retriever.retrieve(query, top_k=self.top_k)
            if hits:
                docs = [hit.doc for hit in hits]
                return format_docs_for_prompt(docs)
            return "未找到相关文档"
        
        tools = [search_knowledge_base]

        # 注意：LangGraph V1.0 已把 create_react_agent 标记为废弃，V2.0 会移除
        # （建议迁移到 langchain.agents.create_agent，需引入完整 langchain 包）。
        # 当前仍可用，先保留；迁移时需一并调整 messages 的提取逻辑。
        return create_react_agent(
            model=self.llm,
            tools=tools,
            prompt="""你是一个智能电商客服Agent，能够帮助用户解答商品、订单、售后、物流等问题。

工作流程：
1. 分析用户问题，确定需要什么知识
2. 使用 search_knowledge_base 工具从知识库检索相关信息
3. 根据检索结果，用专业、友好的语言回答用户

注意：
- 如果知识库没有相关内容，请坦诚告知用户
- 回答要简洁明了，符合客服的专业语气""",
        )

    def answer(
        self,
        question: str,
        session_id: str = "default",
        use_memory: bool = True,
    ):  # type: ignore
        """执行 Agent 回答问题"""
        
        # 获取检索结果（用于评分）
        hits = []
        if self.retriever:
            hits = self.retriever.retrieve(question, top_k=self.top_k)
        
        # 如果没有 LLM，直接降级
        if not self.use_llm or not self.graph:
            return self._fallback_answer(question, hits)
        
        try:
            # 准备消息
            messages = []
            if use_memory and session_id in self._session_memory:
                messages.extend(self._session_memory[session_id])
            messages.append(HumanMessage(content=question))
            
            # 执行 Agent（传入回调以统计 token 用量）
            invoke_config = {"callbacks": self.callbacks} if self.callbacks else {}
            result = self.graph.invoke({"messages": messages}, config=invoke_config)
            
            # 提取结果
            final_messages = result.get("messages", [])
            final_answer = ""
            thoughts = []
            actions = []
            observations = []
            
            for msg in final_messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        # 这是一个思考/行动步骤
                        thoughts.append(msg.content or "正在思考...")
                        for tool_call in msg.tool_calls:
                            actions.append(tool_call.get("name", "tool"))
                    elif msg.content:
                        # 这是最终答案
                        final_answer = msg.content
                elif hasattr(msg, 'name') and msg.name == 'search_knowledge_base':
                    # 工具调用结果
                    observations.append(msg.content)
            
            # 更新记忆
            if use_memory:
                self._update_memory(session_id, messages, final_messages)
            
            # 构建 trace
            trace = {
                "thoughts": thoughts,
                "actions": actions,
                "observations": observations,
                "steps": len(thoughts),
            }
            
            llm_calls = len(thoughts) + 1  # 每次 AIMessage 都是一次 LLM 调用
            
            if not (final_answer or "").strip():
                # 推理模型可能因 token 配额耗尽返回空内容，降级而不是给用户空回答
                print("[LangGraph] empty final answer, falling back")
                return self._fallback_answer(question, hits)

            return final_answer, hits, trace, llm_calls
            
        except Exception as e:
            print(f"[LangGraph] Agent error: {e}")
            return self._fallback_answer(question, hits)

    def _fallback_answer(
        self,
        question: str,
        hits: list[RetrievalHit],
    ):  # type: ignore
        """降级回答"""
        if hits:
            docs = [hit.doc for hit in hits]
            context = format_docs_for_prompt(docs)
            # 简单取第一句作为回答
            first_sentence = context.split("。", 1)[0].replace("\n", " ")
            answer = f"{first_sentence}。"
        else:
            answer = "抱歉，我暂时无法回答这个问题。"
        
        trace = {
            "thoughts": ["降级模式"],
            "actions": ["direct_answer"],
            "observations": [],
            "steps": 1,
        }
        
        return answer, hits, trace, 0

    def _update_memory(
        self,
        session_id: str,
        input_messages: list[BaseMessage],
        output_messages: list[BaseMessage],
    ) -> None:
        """更新会话记忆"""
        # 提取用户的输入和 Agent 的最终输出
        final_answer = None
        for msg in reversed(output_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_answer = msg.content
                break
        
        if not final_answer:
            return
        
        # 获取用户问题
        user_question = ""
        for msg in input_messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
        
        # 更新记忆
        if session_id not in self._session_memory:
            self._session_memory[session_id] = []
        
        memory = self._session_memory[session_id]
        memory.append(HumanMessage(content=user_question))
        memory.append(AIMessage(content=final_answer))
        
        # 控制记忆长度
        max_messages = self.max_memory_messages * 2
        if len(memory) > max_messages:
            memory = memory[-max_messages:]
        
        self._session_memory[session_id] = memory

    # ============================================================
    # 记忆管理
    # ============================================================
    
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
