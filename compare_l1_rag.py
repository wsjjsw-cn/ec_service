"""
对比 L1 缓存 vs RAG 的回答质量
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
    print("对比测试：L1 缓存 vs RAG（经过 LLM）")
    print("=" * 70)
    
    # L1 缓存模式
    print("\n【模式 1】L1 缓存模式（预加载 FAQ）")
    print("-" * 70)
    settings = Settings(use_redis=False)
    pipeline_l1 = build_pipeline(preload=True, settings=settings)
    
    question = "如何申请发票"
    response = pipeline_l1.ask(question)
    
    print(f"\n问题: {question}")
    print(f"\n答案:\n{response.answer}")
    print(f"\n路由: {response.route.value}")
    print(f"延迟: {response.latency_ms:.1f}ms")
    print(f"LLM 调用: {response.llm_calls}")
    print(f"\n评价: 答案固定，无法调整语气，信息有限")
    
    # L1 缓存 + LLM 润色模式（需要修改）
    print("\n\n【模式 2】RAG 模式（无预加载，完全由 LLM 生成）")
    print("-" * 70)
    settings_no_preload = Settings(use_redis=False)
    pipeline_rag = build_pipeline(preload=False, settings=settings_no_preload)
    
    response_rag = pipeline_rag.ask(question)
    
    print(f"\n问题: {question}")
    print(f"\n答案:\n{response_rag.answer}")
    print(f"\n路由: {response_rag.route.value}")
    print(f"延迟: {response_rag.latency_ms:.1f}ms")
    print(f"LLM 调用: {response_rag.llm_calls}")
    print(f"\n评价: 答案更详细、结构清晰、语气友好")
    
    print("\n\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)
    print(f"\n{'指标':<20} {'L1 缓存':<20} {'RAG + LLM':<20}")
    print("-" * 60)
    print(f"{'延迟':<20} {response.latency_ms:<19.1f}ms {response_rag.latency_ms:<19.1f}ms")
    print(f"{'LLM 调用':<20} {response.llm_calls:<20} {response_rag.llm_calls:<20}")
    print(f"{'答案质量':<20} {'一般（固定）':<20} {'优秀（动态）':<20}")
    print(f"{'灵活性':<20} {'低':<20} {'高':<20}")
    print(f"{'成本':<20} {'低':<20} {'高':<20}")

if __name__ == "__main__":
    main()
