"""
路由（Routing）：先用一个 LLM 对输入分类，再通过条件边把请求"路由"到对应的处理节点。
与并行化"全部分支都走"不同，路由是"只走其中一条分支"（互斥分流）。
适用场景：客服意图识别、工单分类分发等"按类型走不同处理逻辑"的场景。

整体流程：
    START -> classify(分类) -> ┬─ refund ─────> handle_refund ─────> END
                               ├─ technical ──> handle_technical ──> END
                               └─ general ────> handle_general ────> END
"""
from typing import TypedDict, Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from models.init_chat_model_llm import deepseek_llm_flash


# 1. 定义状态
# TypedDict 描述了在图节点之间流转的数据结构，每个节点读取并更新其中的字段
class RouteState(TypedDict):
    input: str  # 用户输入的问题（输入）
    category: str  # 用户问题的类型（classify 节点产出）
    response: str  # 回复内容（handle_* 节点产出）


# 2. 定义输出
# 用 Pydantic + Literal 约束 LLM 只能返回固定枚举值，避免自由文本难以路由
class RouteResult(BaseModel):
    category: Literal["refund", "technical", "general"] = Field(description="用户问题的类型")


# 3. 定义结构化输出

# with_structured_output 让大模型强制按 RouteResult 结构返回，可直接拿到 .category
router = deepseek_llm_flash.with_structured_output(RouteResult)


# 4. 定义节点
# 每个节点接收当前 state，返回一个 dict，dict 中的字段会合并（更新）回 state
def classify(state: RouteState) -> dict:
    """分类用户问题的类型"""
    message = router.invoke(f"将用户的请求分类为：refund（退款）,technical（技术问题）,general（一般问题）：{state['input']}")
    return {"category": message.category}

def handle_general(state: RouteState) -> dict:
    """处理一般问题"""
    message = deepseek_llm_flash.invoke(f"请根据用户问题{state['input']}，请以客服的身份做一般性回复")
    return {"response": message.content}

def handle_refund(state: RouteState) -> dict:
    """处理退款问题"""
    message = deepseek_llm_flash.invoke(f"请根据用户问题{state['input']}，请以客服的身份做退款回复")
    return {"response": message.content}

def handle_technical(state: RouteState) -> dict:
    """处理技术问题"""
    message = deepseek_llm_flash.invoke(f"请根据用户问题{state['input']}，请以客服的身份做技术问题回复")
    return {"response": message.content}

def route_by_category(state: RouteState) -> Literal["handle_refund", "handle_technical", "handle_general"]:
     """
     条件边函数：根据分类结果返回下一个目标节点名
     返回值会作为 add_conditional_edges 映射表的 key，决定走哪条分支（互斥，只走一条）
     """
     route_map = {
         "refund": "handle_refund",
         "technical": "handle_technical",
         "general": "handle_general"
     }
     # 找不到对应分类时，默认走 general 分支兜底
     return route_map.get(state.get("category"), "handle_general")



# 5. 定义graph流程
# StateGraph 把上面定义的"状态 + 节点 + 边"组装成一张有向图
graph_builder= StateGraph(RouteState)

# 添加节点：节点名 与 处理函数 绑定
graph_builder.add_node("classify", classify)
graph_builder.add_node("handle_general", handle_general)
graph_builder.add_node("handle_refund", handle_refund)
graph_builder.add_node("handle_technical", handle_technical)

# 添加边
# 入口 -> 分类节点
graph_builder.add_edge(START, "classify")

# 分类节点 -> 条件路由：根据 route_by_category 的返回值，只走三个处理节点中的一个（互斥分流）
graph_builder.add_conditional_edges("classify", route_by_category, {
    "handle_refund": "handle_refund",
    "handle_technical": "handle_technical",
    "handle_general": "handle_general",
})

# 三个处理节点各自直接结束（它们是叶子节点，不会再回到主流程）
graph_builder.add_edge("handle_refund", END)
graph_builder.add_edge("handle_technical", END)
graph_builder.add_edge("handle_general", END)


# 编译图，得到可执行的 Runnable
graph = graph_builder.compile()

# 导出流程图为 PNG 图片（mermaid 渲染）
png_data = graph.get_graph().draw_mermaid_png()
# wb表示二进制写入模式
with open("03_routing.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 03_routing.png")

# 分别用三种不同类型的输入测试路由效果
result1 = graph.invoke({"input":"我的商品被重复扣款，请帮我退款"})
print(result1["response"])
print("="*20)

result2 = graph.invoke({"input":"你们的app在华为手机上不兼容，怎么处理？"})
print(result2["response"])
print("="*20)

result3 = graph.invoke({"input":"你们周六日上班吗？"})
print(result3["response"])
print("="*20)