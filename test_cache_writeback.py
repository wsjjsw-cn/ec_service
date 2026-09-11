"""
验证缓存写回机制
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
    print("验证缓存写回机制")
    print("=" * 70)
    
    # 清空缓存
    settings = Settings(use_redis=False)
    pipeline = build_pipeline(preload=False, settings=settings)  # 不预加载
    
    question = "维修时需要提供什么凭证？"
    
    print(f"\n【第一次查询】: {question}")
    print("-" * 70)
    
    response1 = pipeline.ask(question, write_back=True)
    print(f"路由: {response1.route.value}")
    print(f"缓存等级: {response1.cache_level or '无'}")
    print(f"LLM 调用: {response1.llm_calls}")
    print(f"延迟: {response1.latency_ms:.1f}ms")
    print(f"答案前50字: {response1.answer[:50]}...")
    
    print("\n【第二次查询】: 相同问题")
    print("-" * 70)
    
    response2 = pipeline.ask(question, write_back=True)
    print(f"路由: {response2.route.value}")
    print(f"缓存等级: {response2.cache_level or '无'}")
    print(f"LLM 调用: {response2.llm_calls}")
    print(f"延迟: {response2.latency_ms:.1f}ms")
    
    print("\n【第三次查询】: 相似问题")
    print("-" * 70)
    
    similar_question = "维修需要哪些证明材料？"
    response3 = pipeline.ask(similar_question, write_back=False)
    print(f"问题: {similar_question}")
    print(f"路由: {response3.route.value}")
    print(f"缓存等级: {response3.cache_level or '无'}")
    print(f"LLM 调用: {response3.llm_calls}")
    print(f"延迟: {response3.latency_ms:.1f}ms")
    print(f"答案前50字: {response3.answer[:50]}...")
    
    print("\n" + "=" * 70)
    print("写回验证结果")
    print("=" * 70)
    
    # 验证
    if response1.route.value == "rag" and response2.cache_level:
        print("✅ RAG 结果成功写回缓存！第二次查询命中缓存。")
    elif response2.latency_ms < response1.latency_ms:
        print("✅ 第二次查询明显更快，说明缓存生效！")
    else:
        print("⚠️ 请检查缓存写回逻辑")
    
    print("\n【关键指标对比】")
    print(f"{'指标':<20} {'第一次 (RAG)':<20} {'第二次 (缓存)':<20}")
    print("-" * 60)
    print(f"{'延迟':<20} {response1.latency_ms:<19.1f}ms {response2.latency_ms:<19.1f}ms")
    print(f"{'LLM调用':<20} {response1.llm_calls:<20} {response2.llm_calls:<20}")
    print(f"{'路由':<20} {response1.route.value:<20} {response2.route.value:<20}")

if __name__ == "__main__":
    main()
