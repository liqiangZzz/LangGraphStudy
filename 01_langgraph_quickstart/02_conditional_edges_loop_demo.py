"""
Graph API 计算器 Agent
演示使用 StateGraph 构建一个带工具调用的 Agent ，LLM 决定是否调用工具，工具执行后 LLM 再次推理，循环直到给出最终答案。
"""
import operator
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from models.init_chat_model_llm import deepseek_llm_flash


# ================1. 定义状态 ======================
class CalculatorState(TypedDict):
    """计算器状态，使用 operator.add 归约器自动追加消息"""
    messages: Annotated[list[AnyMessage], operator.add]


# ============ 2. 定义工具 ===========

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


# ============ 3. 创建工具集 ===========
tools = [add, sub, mul, div]

# 创建工具名称映射，把名字作为键，工具本身作为值，存进去
# {"add":add,"sub":sub,"mul":mul,"div":div}
tools_by_name = {tool.name: tool for tool in tools}

# ============ 4. 绑定工具 ===========
model_with_tools = deepseek_llm_flash.bind_tools(tools)


# =================5. 定义节点 ======================
def llm_call(state: CalculatorState) -> dict:
    """
    调用 LLM 返回响应结果。
    若调用失败，返回包含错误信息的 AI 消息，以确保流程继续。
    """
    try:
        response = model_with_tools.invoke(
            [SystemMessage(content="你是一个数学计算助手，请使用工具完成计算并给出答案")] + state['messages']
        )
    except Exception as e:
        # 如果模型调用失败，返回一个错误消息作为 AI 响应
        from langchain_core.messages import AIMessage
        response = AIMessage(content=f"模型调用失败: {e}")
    return {"messages": [response]}


def tool_node(state: CalculatorState) -> dict:
    """
    工具节点：执行 LLM 请求的工具调用。
    对每个工具调用，执行对应的工具，并捕获可能的异常，将结果或错误信息包装成 ToolMessage 返回。
    """
    last_message = state["messages"][-1]

    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        try:
            tool_func = tools_by_name[tool_name]
            result = tool_func.invoke(tool_args)
            # 确保结果是字符串，如果是数字则转为字符串
            content = str(result)
        except Exception as e:
            # 工具执行失败，将错误信息作为 ToolMessage 内容返回
            content = f"工具执行出错: {e}"
        results.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    return {"messages": results}


def should_continue(state: CalculatorState) -> Literal["tool_node", END]:
    """
    检查最后一条消息，如果有 tool_calls 且不为空，则进入工具节点，否则结束。
    增加对消息属性安全访问的判断。
    """
    last_message = state["messages"][-1]

    # 安全地检查是否有 tool_calls 属性且不为空
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END


# ================6. 构建和编译图 ======================
builder_graph = StateGraph(CalculatorState)
# 添加节点
builder_graph.add_node("llm_call", llm_call)
builder_graph.add_node("tool_node", tool_node)

# 添加边
builder_graph.add_edge(START, "llm_call")
# 添加条件边：从 llm_call 节点到 tool_node 节点或 END 节点
builder_graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
# 添加边：工具执行后回到 LLM 以进行下一步推理
builder_graph.add_edge("tool_node", "llm_call")

# ⭐️  这块不应该直接结束，原因：  tool_node 节点可能需要调用多个工具，每次工具调用都会生成一个新的 ToolMessage，
# 所以 tool_node 节点应该一直保持，直到 LLM 不再调用工具为止
#  builder_graph.add_edge("tool_node", END)

agent = builder_graph.compile()

if __name__ == '__main__':
    # 可以更换不同的问题进行测试
    query = "请帮我算一下：(3 + 5) * 2 的结果是多少"
    print(f"用户问题: {query}\n" + "="*50)

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    print("\n完整执行过程:")
    for i, msg in enumerate(result["messages"]):
        print(f"\n--- 消息 {i+1} ---")
        msg.pretty_print()
    print("\n" + "="*50)
    # 提取最终答案（最后一条消息）
    final_message = result["messages"][-1]
    print(f"\n最终答案: {final_message.content}")
