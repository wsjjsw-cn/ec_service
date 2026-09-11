from __future__ import annotations

import os
import json
import requests
from dataclasses import dataclass


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1024


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", "")
        )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if not self.config.api_key:
            return ""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        try:
            response = self._session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM API Error: {e}")
            return ""

    def generate_with_context(self, question: str, context: str) -> str:
        system_prompt = """你是一个专业的电商客服助手。请根据提供的知识库内容，用专业、友好、简洁的语言回答用户问题。
如果知识库中没有相关内容，请坦诚告知用户，不要编造答案。"""
        
        prompt = f"""知识库内容：
{context}

用户问题：{question}

请回答："""
        
        return self.generate(prompt, system_prompt)

    def generate_react(self, question: str, observations: list[str]) -> str:
        system_prompt = """你是一个智能电商客服Agent。遵循ReAct模式（思考-行动-观察）。
现在你已经完成了检索（观察），请根据检索结果生成最终答案。"""
        
        observations_text = "\n".join([f"- {obs}" for obs in observations])
        prompt = f"""用户问题：{question}

检索结果：
{observations_text}

请根据检索结果，用专业、友好的语言回答用户问题。如果检索结果不足以回答，请说明。"""
        
        return self.generate(prompt, system_prompt)
