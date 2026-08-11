"""
工具运行时（ToolRuntime）演示：ToolRuntime + ToolNode 搭配，让工具能读取图状态

案例：电商客服助手。模型通过工具查询订单状态、商品库存后回答用户。
重点不在"调用工具"本身（那是 ToolNode 的事），而在于工具执行时能拿到图状态里的 user_id。

整体流程（ReAct 式循环）：
    START -> llm -> ┬─ 有 tool_calls ─> tools 执行 ─> 回到 llm
                   └─ 无 tool_calls ─> END

为什么搭配 ToolRuntime：
    ToolNode 负责执行工具，但工具函数本身拿不到图状态（如当前 user_id）。
    给工具加一个 runtime: ToolRuntime 参数后，工具内部可通过 runtime.state 读取状态，
    不必把 user_id 塞进 prompt 让模型转述——更安全、更省 token、避免模型泄露或篡改。

核心点：
    - ToolRuntime：注入到工具函数的参数，用 runtime.state 读取图状态；
    - ToolNode(tools)：预置节点，自动执行工具并回传结果；
    - MessagesState：LangGraph 预置状态，自带 messages 字段 + add_messages reducer，
      CustomerState 继承它再扩展 user_id，省去手写消息 reducer。
"""
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolRuntime, ToolNode

from models.init_chat_model_llm import deepseek_llm_flash


# 1. 定义图状态
class CustomerState(MessagesState):
    """在标准消息状态基础上增加 user_id"""
    user_id: str  # 用户Id


# 2. 定义工具
@tool
def check_order(runtime: ToolRuntime, order_id: str) -> str:
    """查询订单状态
     Args:
         order_id: 订单号，如 ORD-001
     Returns:
         订单状态描述
     """
    # 通过 ToolRuntime 从图状态中读取 user_id
    user_id = runtime.state.get("user_id", "unknown")

    mock_orders = {
        "ORD-001": "已发货，预计明天到达",
        "ORD-002": "正在处理中",
        "ORD-003": "已签收",
    }
    status = mock_orders.get(order_id, "未找到该订单")
    return f"用户{user_id}的订单{order_id}：{status}"


@tool
def check_inventory(runtime: ToolRuntime, product_name: str) -> str:
    """查询商品库存
    Args:
        product_name: 商品名称
    Returns:
        商品库存描述
    """
    inventory = {
        "蓝牙耳机": "库存充足（>100件）",
        "机械键盘": "库存紧张（仅剩5件）"
    }
    return inventory.get(product_name, f"未找到商品[{product_name}]")


# 3. 绑定工具到模型
tools = [check_order, check_inventory]
model_with_tools = deepseek_llm_flash.bind_tools(tools)


# 4. 定义 LLM 节点
def llm_node(state: CustomerState) -> dict:
    """LLM 节点：决定调用工具还是直接回答"""
    response = model_with_tools.invoke(
        [SystemMessage(content="你是电商客服助手，用工具查询信息后回答。")]
        + state["messages"]
    )
    return {"messages": [response]}


def should_continue(state: CustomerState) ->  Literal["tools", END]:
    """条件边：有工具调用则进入工具节点，否则结束"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# 5. 定义 graph 流程
graph_builder = StateGraph(CustomerState)

# 添加节点
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", ToolNode(tools))

# 添加边
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", should_continue, ["tools", END])
graph_builder.add_edge("tools", "llm")

graph = graph_builder.compile()

if __name__ == "__main__":
    print("ToolRuntime + ToolNode：电商客服助手")
    print("=" * 60)

    result = graph.invoke({
        "messages": [HumanMessage(content="帮我查一下订单 ORD-001 的状态")],
        "user_id": "user_123",
    })
    print(f"客服: {result['messages'][-1].content}")

    print("-" * 40)

    result = graph.invoke({
        "messages": [HumanMessage(content="蓝牙耳机还有货吗？")],
        "user_id": "user_123",
    })
    print(f"客服: {result['messages'][-1].content}")
