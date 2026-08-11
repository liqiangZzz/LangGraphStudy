"""
工具节点（ToolNode）演示：LLM + ToolNode 搭配，让大模型调用外部工具完成计算

模型本身不擅长精确算术，这里给它注册加减乘除四个工具，由 ToolNode 负责执行：
    模型决定调哪个工具 -> ToolNode 执行 -> 结果回传给模型继续推理，循环直到给出最终答案。

整体流程（ReAct 式循环）：
    START -> llm_call -> ┬─ 有 tool_calls ─> tool_node 执行 ─> 回到 llm_call
                        └─ 无 tool_calls ─> END

为什么搭配 ToolNode：
    它把"解析 tool_calls → 匹配工具 → 执行 → 结果包成 ToolMessage 回传"全自动完成，
    不用手写解析和调用逻辑；支持一条消息多个工具并行执行；异常也会包成 ToolMessage 回传。

核心点：
    - bind_tools：向 LLM 注册工具清单；
    - ToolNode(tools)：预置节点，自动执行工具并回传结果，本案例的核心搭档；
    - should_continue：条件路由，按是否有 tool_calls 决定走工具还是结束；
    - add_messages：messages 的 reducer，追加而非覆盖，保留完整对话历史。
"""
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode

from models.init_chat_model_llm import deepseek_llm_flash


# ============ 1. 定义工具 ===========

@tool
def add(a: int, b: int) -> int:
    """两个整数相加，参数为a和b，返回a+b的结果"""
    return a + b


@tool
def sub(a: int, b: int) -> int:
    """两个整数相减，参数为a和b，返回a-b的结果"""
    return a - b


@tool
def mul(a: int, b: int) -> int:
    """两个整数相乘，参数为a和b，返回a*b的结果"""
    return a * b


@tool
def div(a: int, b: int) -> float:
    """两个整数相除，参数为a和b，返回a/b的结果"""
    return a / b


tools = [add, sub, mul, div]

model_with_tools = deepseek_llm_flash.bind_tools(tools)


# ================2. 定义状态 ======================
class CalculatorState(TypedDict):
    """计算器状态"""
    # messages: Annotated[list[AnyMessage], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]


# =================3. 定义节点 ======================
def llm_call(state: CalculatorState) -> dict:
    """调用LLM进行计算"""
    response = model_with_tools.invoke(
        [SystemMessage(content="你是一个数学计算助手，请使用工具完成计算并给出答案")] + state['messages']
    )

    return {"messages": [response]}


def should_continue(state: CalculatorState) -> Literal["tool_node", END]:
    """判断是否继续使用工具，还是直接返回结果"""
    last_message = state["messages"][-1]
    # 检查是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END


# =================4. 定义 graph 流程 ======================
graph_builder = StateGraph(CalculatorState)

# 添加节点
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tool_node", ToolNode(tools))

# 添加边
graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
graph_builder.add_edge("tool_node", "llm_call")

graph = graph_builder.compile()

# 输出Graph 图/工作流为 PNG
png_data = graph.get_graph().draw_mermaid_png()

with open("calculator_tool_node.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 calculator_tool_node.png")

# ============ 5. 调用图/工作流 ===========

result = graph.invoke({"messages": [HumanMessage(content="帮我计算 （3+5）*2 的结果")]})

# result = agent.invoke({"messages":[{"role":"user","content":"帮我计算 （3+5）*2 的结果"}]})

print("result:", result)

for msg in result["messages"]:
    msg.pretty_print()
