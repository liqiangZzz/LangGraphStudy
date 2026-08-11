"""
评估器-优化器（Evaluator-Optimizer）：一个 LLM 生成内容，另一个 LLM 评估质量，
如果评估不合格，带上反馈重新生成，循环直到达标（或达到最大迭代次数后强制接受）。

业务：生成产品的文案，然后判断文案是否满足要求，不满足继续生成，当重复生成文案3次/满足要求，接受文案

整体流程（带反馈环的循环）：
    START -> 生成文案 -> 评估 -> ┬─ 达标 ─────────────> END
                                ├─ 不达标且迭代<3 ─> 重新生成（带上修改建议）
                                └─ 不达标但已迭代3次 ─> 强制接受，结束
"""
from typing import Literal, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from models.init_chat_model_llm import deepseek_llm_flash


# 1. 定义评估结果的结构
# 用 Pydantic + Literal 约束 LLM 只能返回 "pass" 或 "fail"，避免自由文本难以判断
class ReviewResult(BaseModel):
    grade: Literal["pass", "fail"] = Field(description="文案是否达标，pass=达标，fail=不达标")
    feedback: str = Field(description="不达标时具体的修改建议，达标时返回空字符串")


# 2.  定义结构化输出
# with_structured_output 让大模型强制按 ReviewResult 结构返回，可直接拿到 .grade / .feedback
evaluator = deepseek_llm_flash.with_structured_output(ReviewResult)


# 3.  定义文本状态
# 整个循环过程中流转的数据：每一轮生成和评估都会读取/更新其中的字段
class TextState(TypedDict):
    product: str  # 产品名称（输入）
    draft: str  # 当前文案（generate_text 产出，下一轮会被覆盖更新）
    grade: str  # 评估结论（pass / fail，review_text 产出）
    feedback: str  # 修改建议（review_text 产出，generate_text 下一轮会读取它）
    iteration: int  # 迭代次数（generate_text 每轮 +1，用于限制最大重试次数）


# 4.  定义节点
# 每个节点接收当前 state，返回一个 dict，dict 中的字段会合并（更新）回 state
def generate_text(text_state: TextState) -> dict:
    """生成产品文案，如果有修改建议，根据建议进行修改 """
    # 迭代次数 +1（首轮从 0 -> 1）
    iteration = text_state.get("iteration", 0) + 1
    # 关键：如果有上一轮的 feedback，就带上它和当前草稿一起让 LLM 修改；
    # 否则是首轮，直接生成。这就是"反馈闭环"的体现。
    if text_state.get("feedback"):
        prompt = f"请为产品{text_state['product']}写一条公告，字数30字以内，根据修改建议{text_state['feedback']}，修改当前文案{text_state['draft']}，字数30字以内，要求简洁有力、突出卖点"
    else:
        prompt = f"请为产品{text_state['product']}写一条公告，字数30字以内，要求简洁有力、突出卖点"

    msg = deepseek_llm_flash.invoke(prompt)
    return {"draft": msg.content, "iteration": iteration}


def review_text(text_state: TextState) -> dict:
    """评估文案是否达标，不达标返回修改建议"""
    review_result = evaluator.invoke(
        f"请评估文案是否达标，不达标返回修改建议，达标返回空字符串。文案：{text_state['draft']}")
    return {"grade": review_result.grade, "feedback": review_result.feedback}


def route_review(text_state: TextState) -> str:
    """
    条件路由：根据评估结果决定走向
    返回值用作 add_conditional_edges 映射表的 key：
        - Accepted：达标，或虽不达标但已迭代满 3 次（强制接受，防止死循环）
        - Rejected：不达标且还有重试机会，回到生成器重写
    """
    if text_state["grade"] == "pass":
        return "Accepted"
    elif text_state.get("iteration", 0) >= 3:
        return "Accepted"  # 兜底：避免不达标时无限循环
    else:
        return "Rejected"  # 回到 generate_text，带上 feedback 再生成一次


# 5. 定义工作流
# StateGraph 把上面定义的"状态 + 节点 + 边"组装成一张带反馈环的有向图
graph_builder = StateGraph(TextState)
graph_builder.add_node("generate_text", generate_text)
graph_builder.add_node("review_text", review_text)

# 添加边
# 入口 -> 生成文案
graph_builder.add_edge(START, "generate_text")
# 生成 -> 评估（固定顺序）
graph_builder.add_edge("generate_text", "review_text")
# 评估 -> 条件路由：Rejected 回到生成器形成循环；Accepted 走向结束
# 这条 conditional_edges 就是"反馈环"的来源，让流程可以多轮迭代
graph_builder.add_conditional_edges("review_text", route_review, {"Rejected": "generate_text", "Accepted": END})

# 编译图，得到可执行的 Runnable
graph = graph_builder.compile()

# 输出Graph 图/工作流为 PNG
png_data = graph.get_graph().draw_mermaid_png()
with open("evaluator_optimizer.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 evaluator_optimizer.png")

# 执行图：传入初始 state（只需提供 product，其余字段由节点逐步填充）
# 注意：因含反馈环，内部可能循环多轮，最终 iteration 反映重试了几次
result = graph.invoke({  # type: ignore
    "product":"智能手表"
})
print("result:",result)
print(result["draft"])