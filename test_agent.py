"""
测试 create_react_agent 正确用法
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

def test_agent():
    print("测试 create_react_agent...")
    
    @tool
    def search_knowledge(query: str) -> str:
        """搜索知识库"""
        return f"找到关于 '{query}' 的信息"
    
    # 检查可用参数
    import inspect
    sig = inspect.signature(create_react_agent)
    print(f"参数列表: {list(sig.parameters.keys())}")
    
    # 创建 Agent
    try:
        # 尝试用 state_modifier 代替 system_prompt
        agent = create_react_agent(
            model=None,  # 稍后添加
            tools=[search_knowledge],
            state_modifier="你是一个客服助手",
        )
        print("✓ Agent 创建成功")
    except Exception as e:
        print(f"尝试 state_modifier: {e}")
        
        # 尝试其他方式
        try:
            # 检查是否是 messages_modifier
            agent = create_react_agent(
                model=None,
                tools=[search_knowledge],
                messages_modifier="你是一个客服助手",
            )
            print("✓ Agent 创建成功 (messages_modifier)")
        except Exception as e2:
            print(f"尝试 messages_modifier: {e2}")
            
            # 最简单的调用
            try:
                agent = create_react_agent(
                    model=None,
                    tools=[search_knowledge],
                )
                print("✓ Agent 创建成功 (默认)")
            except Exception as e3:
                print(f"最简单的调用: {e3}")

if __name__ == "__main__":
    test_agent()
