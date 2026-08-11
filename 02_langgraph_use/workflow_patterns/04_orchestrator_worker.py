"""
编排者-工人（Orchestrator-Worker）：一个 LLM 负责把任务动态拆分成若干子任务，
再派发多个 Worker 并行执行，最后汇总成最终结果。
区别于固定并行：子任务数量在运行时才确定（动态 fan-out / fan-in）。

整体流程：
    START -> orchestrator(规划章节) -> [动态 fan-out] -> worker×N(并行写各章节) -> [fan-in] -> synthesize(汇总) -> END

本例最难理解的三个点：
    - Send：用来"动态生成"多个并行 worker 的入口，每个 Send 对应一个 worker 实例；
    - 双状态：主流程用 ReportState，单个 worker 用 WorkerState（只关心自己那一个章节）；
    - operator.add：多个 worker 并行返回结果时，LangGraph 用它把列表自动拼接，而不是互相覆盖。
"""
import operator
from typing import TypedDict, Annotated

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from models.init_chat_model_llm import deepseek_llm_flash

# =========================================================
# 1、定义结构化输出
# =========================================================
class Section(BaseModel):
    """
    Orchestrator 拆分出来的单个章节任务
    """
    title: str = Field(description="章节标题")
    description: str = Field(description="该章节需要完成的内容")


class Sections(BaseModel):
    """
    Orchestrator 最终拆分出的所有章节
    """
    sections: list[Section] = Field(description="多个章节任务")


# ---------------------------------------------------------
# 让 LLM 强制按照 Sections 结构返回
#
# 最终类似：
#
# Sections(
#     sections=[
#         Section(
#             title="AI Agent 基本概念",
#             description="介绍..."
#         ),
#         Section(
#             title="企业应用场景",
#             description="分析..."
#         )
#     ]
# )
# ---------------------------------------------------------

planner = deepseek_llm_flash.with_structured_output(Sections)


# =========================================================
# 2、定义主流程状态
# =========================================================
# ReportState 是"主流程"状态，只在 orchestrator / synthesize 这类全局节点间流转
class ReportState(TypedDict):
    topic: str  # 报告主题（输入）

    # Orchestrator 生成的章节任务
    sections: list[Section]

    # Worker 生成的章节内容
    #
    # Annotated[list[str], operator.add] 是关键：
    # 多个 Worker 并行返回 completed_sections 时，默认行为是"后写覆盖先写"，
    # 加上 operator.add 后，LangGraph 会把所有 Worker 的 list 用 + 拼接合并到一起，
    # 从而实现 fan-in（多个并行结果自动汇总到一个 list）。
    completed_sections: Annotated[list[str], operator.add]

    # 最终完整报告
    final_report: str


# =========================================================
# 3、定义 Worker 状态
# =========================================================
# WorkerState 是"子流程"状态：每个 worker 实例只拿到自己负责的那一个 section，
# 而不是整份 sections，这样 worker 之间互不干扰、可以完全并行。
# 注意 completed_sections 字段名与 ReportState 一致——
# worker 返回的结果会通过 operator.add 合并回主流程的同名字段。
class WorkerState(TypedDict):
    topic: str

    # 当前 Worker 负责的章节（由 assign_workers 通过 Send 传进来）
    section: Section

    # Worker 生成的结果（返回后会合并回 ReportState.completed_sections）
    completed_sections: Annotated[list[str], operator.add]


# =========================================================
# 4、Orchestrator 节点
# =========================================================
def orchestrator(state: ReportState) -> dict:
    """
    根据用户主题，动态拆分报告章节
    """
    result = planner.invoke(
        f"""
    你是一名专业报告规划师。
    现在需要围绕下面的主题生成一篇完整报告：

    主题：{state['topic']}

    请根据主题复杂度，自行决定应该拆分成多少个章节。

    要求：
    1. 章节数量不要固定，根据主题动态决定
    2. 每个章节职责清晰，不要重复
    3. 章节之间应该有合理的逻辑顺序
    4. 每个章节提供 title 和 description
    5. description 要明确告诉写作者这一章应该写什么
    """
    )

    return {
        "sections": result.sections
    }


# =========================================================
# 5、动态分配 Worker
# =========================================================
# 这是"动态 fan-out"的核心：作为 conditional_edges 的路由函数，
# 它返回的不是单个节点名，而是一个 Send 列表——
# 每个 Send("worker", {...}) 都会启动一个 worker 实例，并把第二个参数作为它的初始 state。
# 有几个 section 就发几个 Send，从而实现"运行时才知道并行度"的动态派发。
def assign_workers(state: ReportState):
    """
    根据 orchestrator 生成的 sections，
    动态创建对应数量的 Worker
    """
    return [
        Send(
            "worker",
            {
                # 把主流程的 topic 透传给每个 worker
                "topic": state["topic"],
                # 把当前这一个 section 交给这个 worker（每个 worker 只拿一个）
                "section": section
            }
        )
        for section in state["sections"]
    ]


# =========================================================
# 6、Worker 节点
# =========================================================
def worker(state: WorkerState) -> dict:
    """每个 Worker 只负责写一个章节 """
    section = state["section"]
    msg = deepseek_llm_flash.invoke(f"""
    你是一名专业内容写作者。
    
    报告主题：{state['topic']}
    
    你现在只需要完成其中一个章节。
    
    章节标题： {section.title}
    章节要求：{section.description}
    
    写作要求：
    1. 内容完整、具体
    2. 围绕当前章节展开
    3. 不要写其他章节内容
    4. 使用 Markdown 格式
    5. 章节正文控制在300~500字左右
    6. 不要解释你的写作过程
    
    请直接输出这一章节。 """
     )

    content = f"""
    ## {section.title}
    {msg.content}
    """

    # 返回结果，写回 Orchestrator 主流程
    return {
        "completed_sections": [content]
    }

# =========================================================
# 7、汇总节点
# =========================================================

def synthesize(state: ReportState) -> dict:
    """
    将所有 Worker 的结果合并成最终报告
    """

    all_sections = "\n\n".join(state["completed_sections"])

    msg = deepseek_llm_flash.invoke(f"""
    你是一名专业报告编辑。
    
    下面是多个作者分别完成的报告章节：
    
    {all_sections}
    
    请将这些内容整理成一篇完整报告。
    
    要求：
    1. 保留所有章节的核心内容
    2. 调整章节衔接，使整体逻辑自然
    3. 删除明显重复的表达
    4. 不要大幅删减内容
    5. 使用 Markdown 格式
    6. 不要解释，只输出最终报告
    """
    )

    return {
        "final_report": msg.content
    }


# =========================================================
# 8、创建 LangGraph
# =========================================================
graph_builder = StateGraph(ReportState)

# 添加节点
graph_builder.add_node("orchestrator",orchestrator)
graph_builder.add_node("worker",worker)
graph_builder.add_node("synthesize",synthesize)

# =========================================================
# 9、添加流程
# =========================================================

# START -> orchestrator：先规划章节
graph_builder.add_edge(START,"orchestrator")

# orchestrator -> (动态) worker：
# 这里用 add_conditional_edges，但路由函数 assign_workers 返回的是 Send 列表，
# 所以第三个参数 ["worker"] 只是声明可能去到的节点集合（供图编译时校验/绘制用）。
# 实际运行时走几个 worker、每个 worker 拿什么数据，完全由 assign_workers 返回的 Send 决定。
graph_builder.add_conditional_edges( "orchestrator", assign_workers, ["worker"])

# worker -> synthesize：
# 注意：虽然这里有 N 个 worker 并行，但 LangGraph 会等所有 worker 都完成（隐式 barrier）后，
# 才执行 synthesize；worker 返回的 completed_sections 也已通过 operator.add 合并好。
graph_builder.add_edge( "worker", "synthesize")

# synthesize -> END：汇总后结束
graph_builder.add_edge("synthesize", END)


# =========================================================
# 10、编译
# =========================================================
graph = graph_builder.compile()


# =========================================================
# 11、生成流程图
# =========================================================
png_data = graph.get_graph().draw_mermaid_png()

with open(
    "04_orchestrator_worker.png",
    "wb"
) as f:
    f.write(png_data)

print("流程图已保存到 04_orchestrator_worker.png")


# =========================================================
# 12、运行
# =========================================================
result = graph.invoke({ # type: ignore
        "topic": "人工智能 Agent 在企业自动化中的应用"
    })


# =========================================================
# 13、查看 Orchestrator 拆分的任务
# =========================================================
print("\n========== Orchestrator 拆分结果 ==========\n")
for index, section in enumerate(result["sections"], start=1):
    print(f"{index}. {section.title}")
    print(f"   {section.description}")
    print()


# =========================================================
# 14、查看最终报告
# =========================================================
print("\n========== 最终报告 ==========\n")
print(result["final_report"])