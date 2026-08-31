from langchain.agents import create_agent
from langchain_core.tools import tool

from models.init_chat_model_llm import deepseek_llm_flash


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气为晴朗，25°C。"


# agentx 要和 langgraph.json 中的 agent 名称一致
# 如： "graphs": {
#         "my_agent": "./02_langgraph_use/_langsmith/agent_demo.py:agentx",
#       },
agentx = create_agent(
    model=deepseek_llm_flash,
    tools=[get_weather],
    system_prompt="你是能查询任何问题的助手"
)