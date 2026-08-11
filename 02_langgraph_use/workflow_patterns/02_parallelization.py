"""
并行化（Parallelization）：把一个任务拆成多个互不依赖的子任务同时执行，再由汇总节点收口。
相比串行链式调用，并行可以显著降低整体耗时（墙钟时间≈最慢的那个子任务）。
适用场景：对同一份数据做多维度分析（如本例对评论同时做情感、关键词、垃圾检测）。

整体流程：
    START ──┬──> 情感分析 ──┐
            ├──> 关键词提取 ──┼──> 汇总报告 ──> END
            └──> 垃圾内容检测 ─┘
"""
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from models.init_chat_model_llm import deepseek_llm_flash


# 1. 定义状态结构
# TypedDict 描述了在图节点之间流转的数据结构，每个节点读取并更新其中的字段
class ReviewAnalysisState(TypedDict):
    review: str  # 用户评论原文（输入）
    sentiment: str  # 情感分析结果（analyze_sentiment 节点产出）
    keywords: str  # 关键词提取结果（extract_keywords 节点产出）
    spam_check: str  # 垃圾内容检测结果（check_spam 节点产出）
    report: str  # 汇总报告（aggregate 节点产出）


# 2. 定义节点
# 每个节点接收当前 state，返回一个 dict，dict 中的字段会合并（更新）回 state
def analyze_sentiment(state: ReviewAnalysisState) -> dict:
    """并行任务1 :情感分析节点"""
    message = deepseek_llm_flash.invoke(
        f"请对这条评论进行情感分析，只输出情感倾向（正面、负面、中性），并一句话总结：{state['review']}")
    return {"sentiment": message.content}


def extract_keywords(state: ReviewAnalysisState) -> dict:
    """并行任务2 :关键词提取节点"""
    message = deepseek_llm_flash.invoke(
        f"请从这条评论中提取关键词，用逗号分隔：{state['review']}")
    return {"keywords": message.content}


def check_spam(state: ReviewAnalysisState) -> dict:
    """并行任务3 :垃圾内容检测节点"""
    message = deepseek_llm_flash.invoke(
        f"请检测这条评论是否为垃圾内容，只输出检测结果，不要解释：{state['review']}")
    return {"spam_check": message.content}


def aggregate(state: ReviewAnalysisState) -> dict:
    """汇总节点：把三个并行子任务的结果拼成一份完整报告"""
    report = (
        f"[评论分析报告：]\n"
        f"原文：{state["review"]}\n"
        f"情感分析：{state["sentiment"]}\n"
        f"关键词提取：{state["keywords"]}\n"
        f"垃圾内容检测：{state["spam_check"]}"
    )
    return {"report": report}


# 3. 定义流程
# StateGraph 把上面定义的"状态 + 节点 + 边"组装成一张有向图
graph_builder = StateGraph(ReviewAnalysisState)
# 添加节点：节点名 与 处理函数 绑定
graph_builder.add_node("analyze_sentiment", analyze_sentiment)
graph_builder.add_node("extract_keywords", extract_keywords)
graph_builder.add_node("check_spam", check_spam)
graph_builder.add_node("aggregate", aggregate)

# 添加边：三条 START 出边让三个子任务并行启动（LangGraph 会同时调度它们）
graph_builder.add_edge(START, "analyze_sentiment")
graph_builder.add_edge(START, "extract_keywords")
graph_builder.add_edge(START, "check_spam")

# 合并边到汇总节点：三个并行子任务全部完成后，aggregate 才会被执行（隐式 barrier）
graph_builder.add_edge("analyze_sentiment", "aggregate")
graph_builder.add_edge("extract_keywords", "aggregate")
graph_builder.add_edge("check_spam", "aggregate")
graph_builder.add_edge("aggregate", END)

# 编译图，得到可执行的 Runnable
graph = graph_builder.compile()

# 导出流程图为 PNG 图片（mermaid 渲染）
png_data = graph.get_graph().draw_mermaid_png()
# wb表示二进制写入模式
with open("02_parallelization.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 02_parallelization.png")


if __name__ == "__main__":
    print("并行化模式：商品评论多维分析")
    print("=" * 60)

    review_text = "耳机音质出乎意料的好，降噪效果一流，就是耳套戴久了有点夹耳朵，总体很满意！"
    # 执行图：传入初始 state（只需提供 review，其余字段由节点逐步填充）
    result = graph.invoke({"review": review_text})

    print(result["report"])
