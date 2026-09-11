# 检查 LangChain 可用的类
import sys

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    print("✓ BaseChatModel from langchain_core.language_models.chat_models")
except Exception as e:
    print(f"✗ BaseChatModel: {e}")

try:
    from langchain_core.language_models import BaseChatModel as BCM2
    print("✓ BaseChatModel from langchain_core.language_models")
except Exception as e:
    print(f"✗ BaseChatModel (alt): {e}")

try:
    from langchain_core.runnables.base import Runnable
    print("✓ Runnable")
except Exception as e:
    print(f"✗ Runnable: {e}")

try:
    from langchain_core.prompts import ChatPromptTemplate
    print("✓ ChatPromptTemplate")
except Exception as e:
    print(f"✗ ChatPromptTemplate: {e}")

try:
    from langchain_core.messages import BaseMessage
    print("✓ BaseMessage")
except Exception as e:
    print(f"✗ BaseMessage: {e}")

# 检查 LangGraph
try:
    from langgraph.graph import StateGraph, END
    print("✓ StateGraph, END")
except Exception as e:
    print(f"✗ StateGraph: {e}")

try:
    from langgraph.prebuilt import create_react_agent
    print("✓ create_react_agent")
except Exception as e:
    print(f"✗ create_react_agent: {e}")

# 检查我们的 LLM
try:
    from langchain_deepseek import ChatDeepSeek
    print("✓ ChatDeepSeek")
except Exception as e:
    print(f"✗ ChatDeepSeek: {e}")
    try:
        from langchain_community.chat_models import ChatOpenAI
        print("✓ ChatOpenAI (alternative)")
    except Exception as e2:
        print(f"✗ ChatOpenAI: {e2}")

print("\n=== LangChain Version ===")
try:
    import langchain
    print(f"LangChain: {langchain.__version__}")
except:
    print("Unknown LangChain version")

try:
    import langgraph
    print(f"LangGraph: {langgraph.__version__}")
except:
    print("Unknown LangGraph version")
