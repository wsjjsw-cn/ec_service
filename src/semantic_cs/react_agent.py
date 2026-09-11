from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from semantic_cs.generator import AnswerGenerator
from semantic_cs.models import RetrievalHit
from semantic_cs.retrieval import HybridRetriever

if TYPE_CHECKING:
    from semantic_cs.llm_client import LLMClient


@dataclass
class ReActTrace:
    thoughts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


class ReActRAGAgent:
    def __init__(
        self, 
        retriever: HybridRetriever, 
        generator: AnswerGenerator, 
        llm_client: "LLMClient | None" = None,
        top_k: int = 4
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.llm_client = llm_client
        self.top_k = top_k

    def answer(self, question: str) -> tuple[str, list[RetrievalHit], ReActTrace, int]:
        trace = ReActTrace()
        llm_calls = 1
        
        # Thought 1: Analyze the question
        if self.llm_client:
            thought_prompt = f"用户问题：{question}\n\n请分析这个问题需要什么类型的知识来回答。"
            thought1 = self.llm_client.generate(thought_prompt, "你是一个问题分析专家，用简短的话回答。")
            if thought1:
                trace.thoughts.append(thought1)
            else:
                trace.thoughts.append("判断问题需要知识库检索。")
        else:
            trace.thoughts.append("判断问题需要知识库检索。")
        
        # Action: Retrieve
        trace.actions.append(f"hybrid_retrieve(top_k={self.top_k})")
        hits = self.retriever.retrieve(question, top_k=self.top_k)
        
        # Observation: Record retrieved documents
        if hits:
            obs_text = f"找到 {len(hits)} 个相关文档："
            obs_text += ", ".join([hit.doc.title for hit in hits])
            trace.observations.append(obs_text)
        else:
            trace.observations.append("未找到相关文档")
        
        # Thought 2: Decide if enough info
        if self.llm_client and hits:
            obs_summary = "\n".join([f"【{hit.doc.title}】{hit.doc.body[:100]}" for hit in hits[:2]])
            thought_prompt2 = f"检索结果：\n{obs_summary}\n\n是否有足够信息回答问题？"
            thought2 = self.llm_client.generate(thought_prompt2, "你是一个判断专家，用简短的话回答。")
            if thought2:
                trace.thoughts.append(thought2)
            else:
                trace.thoughts.append("基于检索结果生成答案。")
        else:
            trace.thoughts.append("基于检索结果生成答案。")
        
        # Final Answer: Generate
        answer = self.generator.generate(question, hits)
        
        return answer, hits, trace, llm_calls
