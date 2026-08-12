"""
LangGraph 工作流模式（Workflow Patterns）总结

本目录用 6 个示例演示 LangGraph 中常见的工作流编排模式，
每个文件都是可独立运行的完整示例（运行后会输出对应流程图 PNG）。

通用要素：
    - State：用 TypedDict 定义，在节点间流转的数据，每个节点读取并更新其中字段；
    - Node：一个普通函数 (state)->dict，返回的 dict 会合并回 state；
    - Edge：节点之间的连线，固定边用 add_edge，条件边用 add_conditional_edges；
    - StateGraph：把 State + Node + Edge 组装成有向图，compile() 后得到可执行 Runnable；
    - START / END：图的虚拟起点和终点。

模式清单：
    01_prompt_chaining      提示词链：串行多步骤，上一步输出作为下一步输入
    02_parallelization      并行化：多个互不依赖的子任务同时执行，再汇总
    03_routing              路由：分类后只走其中一条分支（互斥分流）
    03_routing_multi        多分类路由：一次分类出多个类型，并行走多条分支再汇总（路由+并行）
    04_orchestrator_worker  编排者-工人：动态拆分任务 + 并行执行 + 汇总（fan-out/fan-in）
    05_evaluator_optimizer  评估器-优化器：生成-评估-反馈循环，直到达标
"""