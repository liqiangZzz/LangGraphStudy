from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.func import task, entrypoint
from langgraph.graph.message import add_messages  # 标准导入路径

from models.init_chat_model_llm import deepseek_llm_flash


# 1. 定义工具
@tool
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """两个整数相乘"""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """两个整数相除，若除数为零则返回错误描述"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
model_with_tools = deepseek_llm_flash.bind_tools(tools)


# 2. 定义 @task 装饰的任务函数
@task
def call_model(messages: list[BaseMessage]) -> BaseMessage:
    """调用 LLM，返回响应消息；若失败则返回错误消息"""
    try:
        return model_with_tools.invoke(
            [SystemMessage(content="你是一个数学计算助手。请使用工具完成计算并给出最终答案。")]
            + messages
        )
    except Exception as e:
        # 返回一个包含错误信息的 AI 消息，以便 Agent 可以继续或终止
        return AIMessage(content=f"模型调用失败: {e}")


@task
def execute_tool(tool_call: dict) -> ToolMessage:
    """执行单个工具调用，捕获异常并返回错误信息"""
    try:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        return ToolMessage(content=str(result), tool_call_id=tool_call["id"])
    except Exception as e:
        # 返回错误信息作为 ToolMessage，让 LLM 知晓
        return ToolMessage(
            content=f"工具执行出错: {e}",
            tool_call_id=tool_call["id"]
        )

# 3. 定义 @entrypoint 入口函数（Agent 主循环）
@entrypoint()
def calculator_agent(messages: list[BaseMessage]) -> list[BaseMessage]:
    """计算器 Agent：在 while 循环中反复执行 LLM→工具→LLM"""
    model_response = call_model(messages).result()

    while True:
        if not model_response.tool_calls:
            break

        # 并行执行所有工具调用
        tool_result_futures = [
            execute_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [f.result() for f in tool_result_futures]

        # 合并消息历史
        messages = add_messages(messages, [model_response] + tool_results)
        model_response = call_model(messages).result()

    return add_messages(messages, model_response)


# 4. 运行
if __name__ == "__main__":
    final_messages = calculator_agent.invoke(
        [HumanMessage(content="请帮我算一下：100 除以 4 再乘以 3 的结果是多少？")]
    )
    print("--- 完整对话记录 ---")
    for msg in final_messages:
        msg.pretty_print()
