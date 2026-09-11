"""
清除缓存并测试 RAG 的 LLM 使用
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv("src/semantic_cs/.env")

from semantic_cs.bootstrap import build_pipeline
from semantic_cs.config import Settings

def main():
    print("=" * 70)
    print("清除缓存并测试 RAG")
    print("=" * 70)
    
    # 初始化 Pipeline
    settings = Settings(use_redis=False)
    pipeline = build_pipeline(preload=False, settings=settings)
    
    print("\n已清除所有缓存")
    
    # 测试 RAG 问题
    rag_questions = [
        "如何申请发票",
        "维修政策是什么",
        "价格保护怎么用",
    ]
    
    print("\n" + "=" * 70)
    print("RAG 测试（使用 LLM）")
    print("=" * 70)
    
    for question in rag_questions:
        print(f"\n[问题] {question}")
        
        response = pipeline.ask(question)
        
        # 显示前200个字符
        answer = response.answer[:200] + ("..." if len(response.answer) > 200 else "")
        print(f"[回答] {answer}")
        print(f"[路由] {response.route.value}")
        print(f"[LLM调用] {response.llm_calls}")
        print(f"[延迟] {response.latency_ms:.1f}ms")
        
        # 判断是否使用了 LLM
        if response.latency_ms > 100 and response.route.value == "rag":
            print("[状态] ✓ 使用了 LLM 进行 RAG")
        elif response.route.value == "cache":
            print("[状态] ✗ 缓存命中（正常）")
        else:
            print(f"[状态] ??? ({response.route.value})")

if __name__ == "__main__":
    main()
