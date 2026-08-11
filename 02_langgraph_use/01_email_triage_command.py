"""
需求：读取用户邮件，按照邮件紧急程度来进行分类，紧急邮件直接生成回复内容，非紧急邮件先查找知识库搜索相关内容然后再生成回复内容，
对于紧急邮件的回复内容，需要人工进行确认邮件内容，最终生成邮件回复内容。

实现方式：Command
节点 = 数据处理 + 状态更新 + 路由决策
"""
from typing import TypedDict, Literal

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from models.init_chat_model_llm import deepseek_llm_flash


# ======定义大模型返回的结构化输出 ======
class EmailClassification(BaseModel):
    """邮件分类结果"""
    category: Literal["question", "bug", "billing", "other"] = Field(description="邮件类别")
    urgency: Literal["low", "medium", "high"] = Field(description="邮件紧急程度")
    summary: str = Field(description="邮件摘要")


# ===== 定义状态 =====
class EmailState(TypedDict):
    """邮件处理中的状态"""
    sender: str  # 邮件发送者
    email_content: str  # 邮件内容
    classification: dict  # 邮件分类结果
    search_results: list[str]  # 搜索结果
    email_response: str  # 邮件回复内容


classifier = deepseek_llm_flash.with_structured_output(EmailClassification)


# ===== 定义节点 =====
def classify_email(state: EmailState) -> Command[Literal["search_info", "email_reply"]]:
    """邮件分类，大模型进行意图识别进行邮件分类 """
    result = classifier.invoke(f""" 分析一下客户邮件，给出邮件类别、邮件紧急程度、邮件摘要：
           发件人: {state["sender"]},
           邮件内容：{state["email_content"]}""")

    # 将 result 转换为字典
    classification = result.model_dump()
    if classification["urgency"] == "high":
        return Command(
            update={"classification": classification},
            goto="email_reply"
        )

    return Command(
        update={"classification": classification},
        goto="search_info"
    )


def search_info(state: EmailState) -> Command[Literal["email_reply"]]:
    """查找信息，根据邮件内容进行关键词提取，然后在知识库中进行搜索"""
    category = state["classification"]["category"]

    # "question","bug","billing","other"
    data = {
        "question": ["忘记密码请进入设置找到安全输入账号和绑定的手机号进行密码重置"],
        "bug": ["请描述一下你遇到的问题，我们会尽快修复"],
        "billing": ["退款政策：7天内可以申请退款，退款金额为订单金额的80%。"],
        "other": ["请联系我们，我们会尽快处理"],
    }

    result = data.get(category, ["没有查询到相关文档"])

    # 更新 search_results 状态， 跳转到 email_reply 节点
    return Command(
        update={"search_results": result},
        goto="email_reply"
    )


def email_reply(state: EmailState) -> Command[Literal["review_email_reply", END]]:
    """根据搜索结果，生成邮件回复"""
    # 原始邮件
    email_content = state["email_content"]
    # 邮件紧急程度
    urgency = state["classification"]["urgency"]
    # 邮件类别
    category = state["classification"]["category"]
    # 知识库参考
    search_results = state.get("search_results", [])
    knowledge = "\n".join(search_results)

    # 生成邮件回复
    response = deepseek_llm_flash.invoke(f"""
      你是专业的邮件编写助手，根据如下内容来给我草拟回复邮件：
      原始邮件：{email_content}
      邮件紧急程度：{urgency}
      邮件类别：{category}
      知识库参考：
      {knowledge},
      生成的回复邮件要求：语气专业，友好。
    """)

    if urgency == "high":
        return Command(
            update={"email_response": response.content},
            goto="review_email_reply"
        )
    else:
        return Command(
            update={"email_response": response.content},
            goto=END
        )


def review_email_reply(state: EmailState) -> Command[END]:
    """模拟人工审核"""
    return Command(
        update={"email_response": state["email_response"] + "[此内容已经通过人员审核，符合要求]"},
        goto=END
    )


# ===== 定义graph 流程 =====
graph_builder = StateGraph(EmailState)

# 添加节点
graph_builder.add_node("classify_email", classify_email)
graph_builder.add_node("search_info", search_info)
graph_builder.add_node("email_reply", email_reply)
graph_builder.add_node("review_email_reply", review_email_reply)

# 添加边
graph_builder.add_edge(START, "classify_email")

graph = graph_builder.compile()

# 绘制 Mermaid 图
# print(graph.get_graph().draw_ascii()) # 打印 ASCII 图

# 保存 Mermaid 图为 PNG
png_data = graph.get_graph().draw_mermaid_png()

# wb表示二进制写入模式
with open("01_email_triage_command.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 01_email_triage_command.png")

# ====== 使用 graph ======
result1 = graph.invoke({
    "sender": "user@example.com",
    "email_content": "非紧急问题：我的账号密码错误了，我应该怎么处理？",
})

print(result1["email_response"])

print("*" * 20)

result2 = graph.invoke({
    "sender": "user@example.com",
    "email_content": "紧急问题：我的订单中的商品损坏了，我要退款，怎么操作？",
})
print(result2["email_response"])
