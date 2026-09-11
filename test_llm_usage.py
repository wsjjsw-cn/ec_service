"""
快速测试 LLM 使用情况
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
    print("LLM 使用测试")
    print("=" * 70)
    
    # 初始化 Pipeline
    settings = Settings(use_redis=False)
    pipeline = build_pipeline(preload=True, settings=settings)
    
    # 测试用例
    tests = [
        ("你好", "chitchat"),
        ("你是谁", "identity"),
        ("谢谢", "chitchat"),
        ("讲个笑话", "ood"),
        ("七天无理由退货需要什么条件", "l1_cache"),
        ("如何申请发票", "rag"),
    ]
    
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    
    for question, expected_route in tests:
        print(f"\n[问题] {question}")
        print(f"[期望] {expected_route}")
        
        response = pipeline.ask(question)
        
        # 显示前150个字符
        answer = response.answer[:150] + ("..." if len(response.answer) > 150 else "")
        print(f"[回答] {answer}")
        print(f"[路由] {response.route.value}")
        print(f"[LLM调用] {response.llm_calls}")
        print(f"[延迟] {response.latency_ms:.1f}ms")
        
        # 判断是否使用了 LLM
        if response.latency_ms > 50:
            print("[状态] ✓ 使用了 LLM")
        else:
            print("[状态] ✗ 可能未使用 LLM")
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("如果延迟 > 50ms 且回答更自然，说明 LLM 正在工作！")

if __name__ == "__main__":
    main()
