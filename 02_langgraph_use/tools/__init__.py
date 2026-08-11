"""
LangGraph 工具（Tools）模块总结

本目录演示 LangGraph 中让大模型调用外部工具的两种关键搭档，示例位于 node&runtime/ 下，
每个文件都是可独立运行的完整示例。

两个核心预置组件：
    - ToolNode：预置节点，自动完成"解析 tool_calls → 匹配工具 → 执行 → 结果包成 ToolMessage 回传"，
      支持一条消息多个工具并行执行，异常也会包成 ToolMessage 回传，不必手写解析/调用逻辑。
    - ToolRuntime：注入到工具函数的参数，工具内可通过 runtime.state 读取图状态（如 user_id），
      无需把敏感信息塞进 prompt 让模型转述——更安全、更省 token、避免模型泄露或篡改。

示例清单：
    node&runtime/01_calculator_tool_node.py
        LLM + ToolNode 搭配：注册加减乘除工具，让模型调用工具完成精确计算（ReAct 循环）。
    node&runtime/02_ecommerce_tool_runtime_node.py
        ToolRuntime + ToolNode 搭配：电商客服助手，工具执行时通过 ToolRuntime 读取图状态里的 user_id。

通用流程（ReAct 式循环）：
    START -> llm -> ┬─ 有 tool_calls ─> ToolNode 执行 ─> 回到 llm
                   └─ 无 tool_calls ─> END
"""
