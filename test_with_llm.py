"""
测试脚本：演示如何使用真实的 LLM

使用方法：
1. 设置环境变量 DEEPSEEK_API_KEY=your_api_key
2. 运行 python test_with_llm.py

或者修改下方的 settings 参数
"""
from semantic_cs.bootstrap import build_pipeline
from semantic_cs.config import Settings
import time

print("=" * 60)
print("真实 LLM 调用测试")
print("=" * 60)

# 方式1: 从环境变量读取 API Key
# export DEEPSEEK_API_KEY=your_api_key
# export USE_REAL_LLM=true

# 方式2: 直接在代码中设置（不推荐生产使用）
settings = Settings(
    use_redis=False,
    # llm_api_key="your_api_key_here",  # 取消注释并填入你的 API Key
    # use_real_llm=True,
)

pipeline = build_pipeline(settings=settings)

print("\n" + "=" * 60)
print("测试对话")
print("=" * 60)

test_questions = [
    "你好",
    "你是谁",
    "七天无理由退货需要什么条件",
    "如何申请发票",
    "讲个笑话",
]

for question in test_questions:
    print(f"\n用户: {question}")
    start = time.time()
    response = pipeline.ask(question)
    elapsed = (time.time() - start) * 1000
    
    print(f"AI: {response.answer[:100]}{'...' if len(response.answer) > 100 else ''}")
    print(f"路由: {response.route.value} | 耗时: {elapsed:.2f}ms | LLM调用: {response.llm_calls}")
    
    if response.route.value == "rag" and response.debug.get("react"):
        trace = response.debug["react"]
        print(f"  Thought: {trace.get('thoughts', [])[:1]}")
        print(f"  Action: {trace.get('actions', [])[:1]}")
        print(f"  Observation: {trace.get('observations', [])[:1]}")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
