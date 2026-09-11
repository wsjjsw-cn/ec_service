"""
演示脚本：LangChain + LangGraph 完整功能

使用方法：
1. 确保 .env 文件中配置了 DEEPSEEK_API_KEY
2. 运行 python demo_langchain.py
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_cs.bootstrap import build_pipeline
from semantic_cs.config import Settings


def main():
    print("=" * 70)
    print("智能客服系统 - LangChain + LangGraph 演示")
    print("=" * 70)
    
    # 检查是否有 API Key
    from dotenv import load_dotenv
    load_dotenv("src/semantic_cs/.env")
    
    has_api_key = bool(os.getenv("DEEPSEEK_API_KEY"))
    print(f"\n[配置] LLM API Key: {'已配置' if has_api_key else '未配置 (降级模式)'}")
    print(f"[配置] 模型: {os.getenv('DEEPSEEK_MODEL', 'default')}")
    print("=" * 70)
    
    # 初始化 Pipeline
    settings = Settings(
        use_redis=False,  # 使用内存存储
    )
    
    pipeline = build_pipeline(preload=True, settings=settings)
    
    print("\n" + "=" * 70)
    print("开始测试对话")
    print("=" * 70)
    
    # 测试用例
    test_cases = [
        # 基本功能
        ("你好", "default"),
        ("你是谁", "default"),
        ("谢谢", "default"),
        
        # 业务问题 (L1 缓存)
        ("七天无理由退货需要什么条件", "default"),
        
        # 业务问题 (RAG)
        ("如何申请发票", "default"),
        ("维修政策是什么", "default"),
        
        # 超出范围
        ("讲个笑话", "default"),
        ("今天天气怎么样", "default"),
        
        # 多轮对话测试记忆
        ("我想退货", "user123"),
        ("需要什么条件", "user123"),  # 应该利用上下文
        ("多久能退回来", "user123"),  # 应该利用上下文
        
        # 另一个用户的独立会话
        ("你好，我要换货", "user456"),
        ("换货有什么要求", "user456"),
    ]
    
    print()
    
    for question, session_id in test_cases:
        print(f"\n[用户{session_id}] {question}")
        
        try:
            response = pipeline.ask(
                question,
                session_id=session_id,
                use_memory=True,
            )
            
            # 只显示前200个字符
            answer = response.answer[:200] + ("..." if len(response.answer) > 200 else "")
            print(f"[AI] {answer}")
            print(f"     路由: {response.route.value} | 缓存: {response.cache_level or '无'} | LLM调用: {response.llm_calls} | 延迟: {response.latency_ms:.1f}ms")
            
            # 显示 RAG 的思考过程
            if response.route.value == "rag" and response.debug:
                debug = response.debug
                if "steps" in debug:
                    print(f"     RAG 步骤数: {debug['steps']}")
                
        except Exception as e:
            print(f"[错误] {e}")
    
    print("\n" + "=" * 70)
    print("会话记忆演示")
    print("=" * 70)
    
    # 演示会话记忆
    print("\n--- user123 的对话历史 ---")
    memory = pipeline.get_session_memory("user123")
    print(f"记忆条数: {len(memory)}")
    for msg in memory[:4]:  # 只显示前4条
        role = "用户" if "HumanMessage" in str(type(msg)) else "AI"
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  [{role}] {content}")
    
    print("\n--- user456 的对话历史 ---")
    memory = pipeline.get_session_memory("user456")
    print(f"记忆条数: {len(memory)}")
    for msg in memory[:2]:
        role = "用户" if "HumanMessage" in str(type(msg)) else "AI"
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  [{role}] {content}")
    
    print("\n" + "=" * 70)
    print("清理会话")
    print("=" * 70)
    
    pipeline.clear_all_memory()
    print("✓ 已清理所有会话记忆")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
