"""
构建简单的Graph 图/工作流：
    构建一个节点（大模型节点）、两条边（START → call_model → END）连接起来整个流程的工作流。
核心概念：
    1. StateGraph 必须创建并编译后才能 invoke。
    2. 状态使用 MessagesState（包含 messages 字段，存储对话历史），默认使用 MessagesState，用户可以自定义其他状态类。
    3. 节点对应 Python 函数，参数为状态，节点之间通过边连接起来，返回更新后的状态片段 。
    4. add_node：添加节点，参数为节点名称和节点函数；add_edge：添加边，参数为源节点和目标节点。
    5. 用户消息支持 HumanMessage 对象或 {"role":"user","content":"..."} 字典。
    6. 用户传入的消息必须符合 MessagesState 中定义的字段，否则会报错。
    7. 节点返回的 {"messages": [response]} 会通过 add_messages 归约器追加到已有消息列表。

"""
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

from models.init_chat_model_llm import deepseek_llm_flash


def call_model(state: MessagesState):
    """调用大模型，将返回消息追加到状态中"""
    print("state:", state)
    try:
        response = deepseek_llm_flash.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"服务暂时不可用，错误: {e}")
        return {"messages": [error_msg]}


def build_simple_graph():
    """创建一个单节点 LangGraph"""

    # 创建一个 StateGraph 实例，用于构建 LangGraph
    graph_builder = StateGraph(MessagesState)

    # 添加节点：call_model 节点，用于调用大模型
    graph_builder.add_node("call_model", call_model)

    # 添加边：从 START 节点到 call_model 节点
    graph_builder.add_edge(START, "call_model")

    # 添加边：从 call_model 节点到 END 节点
    graph_builder.add_edge("call_model", END)

    return graph_builder.compile()


if __name__ == '__main__':
    graph = build_simple_graph()
    result =  graph.invoke({"messages":[{"role": "user", "content": "你好,请介绍下你自己"}]})
    # result = graph.invoke({"messages": [HumanMessage(content="你好,请介绍下你自己")]})
    print("result:", result)

    print("结果:")
    for msg in result["messages"]:
        msg.pretty_print()
