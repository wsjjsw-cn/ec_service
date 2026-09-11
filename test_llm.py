"""
测试 LLM 是否正常工作
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv("src/semantic_cs/.env")

print("=" * 60)
print("检查环境变量")
print("=" * 60)
print(f"DEEPSEEK_API_KEY: {'已设置' if os.getenv('DEEPSEEK_API_KEY') else '未设置'}")
print(f"DEEPSEEK_BASE_URL: {os.getenv('DEEPSEEK_BASE_URL', '未设置')}")
print(f"DEEPSEEK_MODEL: {os.getenv('DEEPSEEK_MODEL', '未设置')}")

print("\n" + "=" * 60)
print("测试 LLM 连接")
print("=" * 60)

try:
    from langchain_deepseek import ChatDeepSeek
    
    llm = ChatDeepSeek(
        model=os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash'),
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
        temperature=0.7,
    )
    
    print("✓ LLM 对象创建成功")
    
    print("\n测试简单对话...")
    response = llm.invoke("你好，请用一句话介绍你自己")
    print(f"✓ LLM 响应: {response.content}")
    
    print("\n测试系统提示词...")
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content="你是一个专业的电商客服助手，名叫小智。"),
        HumanMessage(content="你是谁？"),
    ]
    
    response = llm.invoke(messages)
    print(f"✓ LLM 响应: {response.content}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！LLM 正常工作")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ LLM 连接失败: {e}")
    import traceback
    traceback.print_exc()
