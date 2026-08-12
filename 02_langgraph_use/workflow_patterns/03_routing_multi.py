"""
多分类路由（Routing Multi）：先分类出"多个"类型，再通过条件边同时路由到多个处理节点并行执行，最后汇总。

与 03_routing 的区别：
    03_routing 是"单分类 + 单分支"——一条输入只归一类，只走一条分支（互斥分流）。
    本例是"多分类 + 多分支"——一条输入可同时属于多个类型（如"重复扣款 + APP 无法登录"
    同时是 refund 和 technical），条件边一次返回多个节点名，LangGraph 会并行执行它们，
    再由 aggregate 汇总。本质是路由 + 并行的组合。

关键改动点（相对 03_routing）：
    - 分类结果从单个 category 改为列表 categories（LLM 一次可返回多个类型）；
    - 条件边函数 route_by_category 返回的是"节点名列表"而非单个字符串，从而触发多分支并行；
    - 状态用 Annotated[list[str], operator.add] 收集多个处理节点的结果（reducer 自动拼接）；
    - 新增 aggregate 汇总节点，把多个回复拼成一份最终输出。

整体流程：
    START -> classify -> ┬─ refund ─────> handle_refund ──┐
                         ├─ technical ──> handle_technical ┼─> aggregate -> END
                         └─ general ────> handle_general ──┘
    （命中几条就并行走几条，全部完成后汇总）
"""
import operator
from typing import TypedDict, Literal, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from models.init_chat_model_llm import deepseek_llm_flash


# ============================
# 1. 定义 State
# ============================
class RouteState(TypedDict):
    input: str  # 用户输入的问题（输入）
    categories: list[str]  # 用户问题的类型（classify 节点产出）
    # 多个节点返回结果
    # 使用 Reducer 合并
    responses: Annotated[list[str], operator.add]  # 回复内容（handle_* 节点产出）


# ============================
# 2. 定义 LLM 输出结构
# ============================
class RouteResult(BaseModel):
    categories: list[
        Literal[
            "refund",
            "technical",
            "general"
        ]
    ] = Field(
        description="""
          用户问题涉及的所有类型：
          refund：退款问题
          technical：技术问题
          general：一般问题
          """
    )


# ============================
# 3. 创建结构化输出模型
# ============================
router = deepseek_llm_flash.with_structured_output(RouteResult)


# ============================
# 4. 定义节点
# ============================
def classify(state: RouteState) -> dict:
    """
    分类节点

    判断用户问题属于哪些类型
    可以返回多个分类
    """
    result = router.invoke(
        f"""
        判断用户问题涉及哪些类型：

        refund：
        退款、订单扣款、支付问题

        technical：
        APP异常、系统问题、技术问题

        general：
        普通咨询问题


        用户问题：
        {state["input"]}
        """
    )
    return {"categories": result.categories}


def handle_refund(state: RouteState):
    """
    退款处理节点
    """
    result = deepseek_llm_flash.invoke(
        f"""
        用户问题：
        {state["input"]}

        请作为客服处理退款问题。
        """
    )
    return {"responses": [result.content]}


def handle_technical(state: RouteState):
    """
    技术问题处理节点
    """
    result = deepseek_llm_flash.invoke(
        f"""
        用户问题：
        {state["input"]}

        请作为技术客服解决问题。
        """
    )
    return {"responses": [result.content]}


def handle_general(state: RouteState):
    """
    一般问题处理节点
    """
    result = deepseek_llm_flash.invoke(
        f"""
        用户问题：
        {state["input"]}

        请作为客服进行一般咨询回复。
        """
    )
    return {"responses": [result.content]}


# ============================
# 5. 条件路由函数
# ============================
def route_by_category(state: RouteState):
    """
    条件边函数
    返回多个节点名称
    LangGraph 会同时执行多个节点
    """

    route_map = {
        "refund": "handle_refund",
        "technical": "handle_technical",
        "general": "handle_general"
    }
    nodes = []
    for category in state["categories"]:
        if category in route_map:
            nodes.append(route_map[category])

    # 默认兜底
    if not nodes:
        nodes.append("handle_general")
    return nodes


# ============================
# 6. 汇总节点
# ============================
def aggregate(state: RouteState):
    """
    汇总多个节点结果
    """
    final_response = "\n\n".join(state["responses"])
    return {"responses": [final_response]}


# ============================
# 7. 创建 Graph
# ============================
graph_builder = StateGraph(RouteState)

# 添加节点：节点名 与 处理函数 绑定
graph_builder.add_node("classify", classify)
graph_builder.add_node("handle_refund", handle_refund)
graph_builder.add_node("handle_technical", handle_technical)
graph_builder.add_node("handle_general", handle_general)
graph_builder.add_node("aggregate", aggregate)

# ============================
# 添加边
# ============================
graph_builder.add_edge(START, "classify")

# classify -> 多节点路由
graph_builder.add_conditional_edges("classify", route_by_category, {
    "handle_refund": "handle_refund",
    "handle_technical": "handle_technical",
    "handle_general": "handle_general",
})

# 多个处理节点 -> aggregate
graph_builder.add_edge("handle_refund", "aggregate")
graph_builder.add_edge("handle_technical", "aggregate")
graph_builder.add_edge("handle_general", "aggregate")

# aggregate -> END
graph_builder.add_edge("aggregate", END)

# ============================
# 8. 编译
# ============================
graph = graph_builder.compile()

# ============================
# 9. 导出流程图
# ============================
png_data = graph.get_graph().draw_mermaid_png()
# wb表示二进制写入模式
with open("03_routing_multi.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 03_routing_multi.png")

# ============================
# 10. 测试
# ============================
result = graph.invoke({  # type: ignore
    "input": "我的订单被重复扣款，而且APP无法登录",
    "responses": []
})

print("分类结果：", result["categories"])
print("\n最终回复：")
print(result["responses"][0])
