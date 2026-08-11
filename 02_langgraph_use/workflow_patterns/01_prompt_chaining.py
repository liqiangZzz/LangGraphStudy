"""
提示词链（Prompt Chaining）：把一个复杂任务拆成多个串联的小步骤，每个步骤由一次 LLM 调用完成，
上一步的输出作为下一步的输入，依次推进，中间可通过条件边决定走向。

整体流程：
    生成初稿 -> 检查字数 -> 不足则扩写 -> 润色 -> 输出最终文案
    生成初稿 -> 检查字数 -> 足够则直接作为最终文案
"""
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from models.init_chat_model_llm import deepseek_llm_flash


# 1、 ======== 定义状态 ===========
# TypedDict 描述了在图节点之间流转的数据结构，每个节点读取并更新其中的字段
class CopywritingState(TypedDict):
    product: str  # 产品名称（输入）
    draft: str  # 初稿文案（generate_draft 节点产出）
    expanded: str  # 扩写文案（expand_draft 节点产出）
    final: str  # 最终文案（polish 节点产出，或初稿直接作为最终结果）


# 2、 ======== 定义节点 ===========
# 每个节点接收当前 state，返回一个 dict，dict 中的字段会合并（更新）回 state
def generate_draft(state: CopywritingState) -> dict:
    """
    节点1：根据产品名称生成初稿文案
    """
    msg = deepseek_llm_flash.invoke(f"""
    请为产品“{state['product']}”生成一段营销文案初稿。

    要求：
    1. 突出产品的核心卖点
    2. 语言自然、有吸引力
    3. 可以包含简单的使用场景
    4. 不要解释，不要标题，只输出文案本身
    """
                                    )

    draft = msg.content

    return {
        "draft": draft,
        # 默认把初稿作为最终结果
        # 如果后续经过 polish，会被覆盖
        "final": draft
    }


def expand_draft(state: CopywritingState) -> dict:
    """
    节点2：初稿不足40字时扩写，根据初稿文案生成扩写文案
    """
    msg = deepseek_llm_flash.invoke(f"""
    下面是一段较短的产品营销文案：
    {state['draft']}
    请将它扩写到60字左右。
    要求：
    1. 保留原文的核心意思
    2. 补充产品的核心卖点
    3. 增加具体的使用场景
    4. 突出用户能够获得的价值
    5. 不要解释，只输出扩写后的文案
    """
                                    )
    return {"expanded": msg.content}


def polish_draft(state: CopywritingState) -> dict:
    """
    节点3：根据扩写文案生成最终文案,润色文案
    """
    msg = deepseek_llm_flash.invoke(f"""
    请润色下面这段产品营销文案：

    {state['expanded']}

    要求：
    1. 保持原有卖点和信息不变
    2. 语言更加自然、有感染力
    3. 减少生硬和重复表达
    4. 更符合营销文案的表达方式
    5. 保持60字左右
    6. 不要解释，只输出最终文案
    """
                                    )
    return {"final": msg.content}


# 3、 ======== 定义条件 ===========
# 条件函数：返回值用作 add_conditional_edges 映射表的 key，决定下一步走向哪个节点
def check_length(state: CopywritingState) -> str:
    """
    检查文案长度是否超过40个字
    """
    if len(state["draft"]) >= 40:
        print("文案长度超过40个字，使用初稿作为最终文案")
        return "Pass"  # 字数足够，直接结束
    else:
        return "Fall"  # 字数不足，需要扩写+润色


# 4、 ======== 定义流程 ===========
# StateGraph 把上面定义的"状态 + 节点 + 边"组装成一张有向图
graph_builder = StateGraph(CopywritingState)

# 添加节点：节点名 与 处理函数 绑定
graph_builder.add_node("generate_draft", generate_draft)
graph_builder.add_node("expand_draft", expand_draft)
graph_builder.add_node("polish_draft", polish_draft)

# 添加边：描述节点之间的执行顺序
# 入口 -> 生成初稿
graph_builder.add_edge(START, "generate_draft")
# 生成初稿后，由 check_length 的返回值决定走向：字数不足则扩写，足够则直接结束
graph_builder.add_conditional_edges("generate_draft", check_length, {"Fall": "expand_draft", "Pass": END})
# 扩写 -> 润色
graph_builder.add_edge("expand_draft", "polish_draft")
# 润色 -> 结束
graph_builder.add_edge("polish_draft", END)

# 编译图，得到可执行的 Runnable
graph = graph_builder.compile()

# 导出流程图为 PNG 图片（mermaid 渲染）
png_data = graph.get_graph().draw_mermaid_png()
# wb表示二进制写入模式
with open("01_prompt_chaining.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 01_prompt_chaining.png")

# 执行图：传入初始 state（只需提供 product，其余字段由节点逐步填充）
result = graph.invoke({  # type: ignore
    "product": "智能手环"
})
print("result:", result)

# 输出最终文案（可能来自初稿，也可能来自润色节点）
print(result["final"])
