# LangGraphStudy

LangGraph 学习与实践项目。通过一系列由浅入深、可独立运行的示例，演示 LangGraph 的核心概念（State / Node / Edge / StateGraph）以及常见工作流编排模式、工具调用、LangSmith 集成等进阶用法。

所有示例统一使用 DeepSeek（`deepseek-v4-pro` / `deepseek-v4-flash`）与 GLM（`glm-5.1`）模型，通过 `models/init_chat_model_llm.py` 共享模型实例。

## 目录结构

```
LangGraphStudy/
├── 01_langgraph_quickstart/        # 快速入门：从单节点到带工具循环的 Agent
│   ├── 01_simple_langgraph_single_node.py      # 单节点 Graph（START → call_model → END）
│   ├── 02_conditional_edges_loop_demo.py       # 条件边 + 工具循环 Agent（StateGraph）
│   └── 03_functional_api_calculator_agent.py   # Functional API（task/entrypoint）计算器 Agent
│
├── 02_langgraph_use/               # 进阶用法
│   ├── 01_email_triage_command.py             # 邮件分类：Command 实现路由
│   ├── 02_email_triage_conditional_edges_list.py  # 邮件分类：条件边 + 列表（返回真实节点名）
│   ├── 03_email_triage_conditional_edges_dict.py  # 邮件分类：条件边 + 字典（业务标签映射节点）
│   ├── workflow_patterns/                     # 5 种工作流编排模式
│   │   ├── 01_prompt_chaining.py              # 提示词链：串行多步骤
│   │   ├── 02_parallelization.py              # 并行化：多子任务同时执行再汇总
│   │   ├── 03_routing.py                      # 路由：分类后互斥分流
│   │   ├── 04_orchestrator_worker.py          # 编排者-工人：动态 fan-out/fan-in
│   │   └── 05_evaluator_optimizer.py          # 评估器-优化器：生成-评估-反馈循环
│   ├── tools/                                 # 工具调用（ToolNode / ToolRuntime）
│   │   └── node&runtime/
│   │       ├── 01_calculator_tool_node.py     # LLM + ToolNode：计算器
│   │       └── 02_ecommerce_tool_runtime_node.py  # ToolRuntime + ToolNode：电商客服
│   └── langsmith/                             # LangSmith / langgraph dev server 集成
│       ├── agent_demo.py                      # create_agent 创建 Agent（注册为 my_agent）
│       └── graph_client.py                    # SDK 客户端流式调用 graph
│
├── models/
│   └── init_chat_model_llm.py                 # 共享模型实例（deepseek_llm_pro/flash、glm_llm）
├── env_utils.py                               # 加载 .env，导出 API Key / Base URL
├── .env.example                               # 环境变量模板
├── langgraph.json                             # langgraph dev server 配置（注册 graph/agent）
└── pyproject.toml                             # 项目打包配置（setuptools）
```

## 核心概念速览

| 概念 | 说明 |
|------|------|
| **State** | 用 `TypedDict` 定义，在节点间流转的数据，每个节点读取并更新其中字段 |
| **Node** | 一个普通函数 `(state) -> dict`，返回的 dict 会合并回 state |
| **Edge** | 节点之间的连线；固定边用 `add_edge`，条件边用 `add_conditional_edges` |
| **StateGraph** | 把 State + Node + Edge 组装成有向图，`compile()` 后得到可执行 Runnable |
| **START / END** | 图的虚拟起点和终点 |
| **ToolNode** | 预置节点，自动完成"解析 tool_calls → 执行 → 回传 ToolMessage" |
| **ToolRuntime** | 注入工具函数的参数，工具内可读取图状态（如 user_id） |

## 环境准备

1. 安装依赖（推荐 Python ≥ 3.10）：

```bash
pip install -e .
```

2. 复制环境变量模板并填写真实 API Key：

```bash
cp .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / GLM_API_KEY / GLM_BASE_URL / LANGSMITH_API_KEY
```

## 运行示例

每个示例文件均可独立运行，例如：

```bash
python 02_langgraph_use/workflow_patterns/01_prompt_chaining.py
python 02_langgraph_use/tools/node&runtime/01_calculator_tool_node.py
```

运行后会在当前目录生成对应的流程图 PNG（mermaid 渲染）。

## 使用 langgraph dev server

本项目已通过 `langgraph.json` 注册了两个 graph，可用 LangGraph 开发服务器调试：

```bash
langgraph dev
```

启动后（默认 `http://127.0.0.1:2024`），可通过 `02_langgraph_use/langsmith/graph_client.py` 用 SDK 客户端流式调用：

```bash
python 02_langgraph_use/langsmith/graph_client.py
```

`langgraph.json` 中注册的内容：

| 名称 | 指向 | 说明 |
|------|------|------|
| `my_graph` | `tools/node&runtime/01_calculator_tool_node.py:graph` | 计算器工具图 |
| `my_agent` | `langsmith/agent_demo.py:agentx` | create_agent 创建的带工具 Agent |

## 工作流模式一览

`02_langgraph_use/workflow_patterns/` 下的 5 个模式：

| 模式 | 一句话说明 |
|------|-----------|
| **提示词链** (Prompt Chaining) | 把复杂任务拆成多个串联小步骤，上一步输出作为下一步输入 |
| **并行化** (Parallelization) | 拆成互不依赖的子任务同时执行，再由汇总节点收口 |
| **路由** (Routing) | 先分类，再通过条件边只走其中一条分支（互斥分流） |
| **编排者-工人** (Orchestrator-Worker) | 动态拆分任务 + 并行执行 + 汇总（fan-out / fan-in） |
| **评估器-优化器** (Evaluator-Optimizer) | 生成-评估-反馈循环，直到达标 |

各文件头均有详细中文注释（模式定义 + 流程图 + 核心点说明）。
