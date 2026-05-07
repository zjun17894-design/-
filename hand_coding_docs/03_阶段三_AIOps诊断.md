# 阶段三：AIOps 智能诊断

## 本阶段目标

实现基于 Plan-Execute-Replan 模式的智能故障诊断系统，包括：
- AIOps 状态定义
- Planner 节点（制定计划）
- Executor 节点（执行步骤）
- Replanner 节点（重新规划/生成响应）
- AIOps 服务
- 文件上传接口
- 静态文件托管

---

## 文件编写顺序

```
1. app/models/aiops.py → AIOps 请求模型
2. app/agent/aiops/state.py → 状态定义
3. app/agent/aiops/utils.py → 工具函数
4. app/agent/aiops/__init__.py → 模块导出
5. app/agent/aiops/planner.py → Planner 节点
6. app/agent/aiops/executor.py → Executor 节点
7. app/agent/aiops/replanner.py → Replanner 节点
8. app/services/aiops_service.py → AIOps 服务
9. app/api/aiops.py → AIOps 接口
10. app/api/file.py → 文件上传接口
11. 修改 main.py → 添加路由和静态文件
```

---

## 1. app/models/aiops.py - AIOps 请求模型

### 文件作用
定义 AIOps 诊断的请求模型。

### 手敲代码

```python
# app/models/aiops.py
from pydantic import BaseModel, Field


class AIOpsRequest(BaseModel):
    """AIOps 诊断请求"""
    session_id: str = Field(..., description="会话 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-123"
            }
        }
```

---

## 2. app/agent/aiops/state.py - 状态定义

### 文件作用
定义 Plan-Execute-Replan 工作流的状态结构。

### 手敲代码

```python
# app/agent/aiops/state.py
from typing import List, TypedDict, Annotated
import operator


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""

    # 用户输入（任务描述）
    input: str

    # 执行计划（步骤列表）
    plan: List[str]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新
    past_steps: Annotated[List[tuple], operator.add]

    # 最终响应/报告
    response: str
```

### 调用关系
- **被谁调用**：planner.py, executor.py, replanner.py, aiops_service.py
- **调用谁**：无

### 初学者最容易误解的地方
1. **Annotated[List, operator.add]**：表示 `past_steps` 字段更新时是追加而非替换
2. **TypedDict 的作用**：定义字典的键和值类型，让代码更清晰
3. **tuple**：`past_steps` 的元素是元组 `(step, result)`

---

## 3. app/agent/aiops/utils.py - 工具函数

### 文件作用
提供工具相关的辅助函数。

### 手敲代码

```python
# app/agent/aiops/utils.py
from typing import List, Any


def format_tools_description(tools: List[Any]) -> str:
    """格式化工具列表为描述文本

    Args:
        tools: 工具列表

    Returns:
        str: 格式化的工具描述
    """
    if not tools:
        return "无可用工具"

    descriptions = []
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        description = getattr(tool, "description", "")

        if description:
            descriptions.append(f"- {name}: {description}")
        else:
            descriptions.append(f"- {name}")

    return "\n".join(descriptions)
```

---

## 4. app/agent/aiops/__init__.py - 模块导出

```python
# app/agent/aiops/__init__.py
from app.agent.aiops.state import PlanExecuteState
from app.agent.aiops.planner import planner
from app.agent.aiops.executor import executor
from app.agent.aiops.replanner import replanner

__all__ = ["PlanExecuteState", "planner", "executor", "replanner"]
```

---

## 5. app/agent/aiops/planner.py - Planner 节点

### 文件作用
制定执行计划节点，负责将用户任务分解为可执行的步骤。

### 手敲代码

```python
# app/agent/aiops/planner.py
from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from app.agent.aiops.state import PlanExecuteState
from app.agent.aiops.utils import format_tools_description


class Plan(BaseModel):
    """计划的输出格式"""
    steps: List[str] = Field(
        description="完成任务所需的不同步骤"
    )


planner_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

        可用工具列表：
        {tools_description}

        注意：你的职责是制定计划，实际的工具调用由 Executor 负责。

        对于给定的任务，请创建一个简单的、逐步的计划来完成它：
        - 将任务分解为逻辑上独立的步骤
        - 每个步骤应该明确使用哪些工具来获取信息
        - 步骤之间应该有清晰的依赖关系
    """).strip()),
    ("placeholder", "{messages}"),
])


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """规划节点：根据用户输入生成执行计划"""
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")

    # 获取本地工具
    local_tools = [get_current_time, retrieve_knowledge]

    # 获取 MCP 工具
    mcp_client = await get_mcp_client_with_retry()
    mcp_tools = await mcp_client.get_tools()

    # 合并所有工具
    all_tools = local_tools + mcp_tools
    tools_description = format_tools_description(all_tools)

    # 创建 LLM
    llm = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0
    )

    planner_chain = planner_prompt | llm.with_structured_output(Plan)

    # 调用 LLM 生成计划
    plan_result = await planner_chain.ainvoke({
        "messages": [("user", input_text)],
        "tools_description": tools_description
    })

    # 提取步骤列表
    if isinstance(plan_result, Plan):
        plan_steps = plan_result.steps
    else:
        plan_steps = plan_result.get("steps", [])

    logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
    for i, step in enumerate(plan_steps, 1):
        logger.info(f"  步骤{i}: {step}")

    return {"plan": plan_steps}
```

### 调用关系
- **被谁调用**：aiops_service.py (LangGraph 工作流)
- **调用谁**：ChatQwen, tools, mcp_client

### 初学者最容易误解的地方
1. **planner_prompt 的作用**：这是系统提示词，告诉 LLM 如何制定计划
2. **with_structured_output(Plan)**：强制 LLM 输出符合 Plan 模型的结构化数据
3. **返回值格式**：`{"plan": [步骤列表]}`，这会更新状态中的 plan 字段

---

## 6. app/agent/aiops/executor.py - Executor 节点

### 文件作用
执行计划中的单个步骤，使用 LangGraph 的 ToolNode 自动处理工具调用。

### 手敲代码

```python
# app/agent/aiops/executor.py
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from app.agent.aiops.state import PlanExecuteState


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """执行节点：执行计划中的下一个步骤"""
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        # 获取本地工具
        local_tools = [get_current_time, retrieve_knowledge]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建 LLM（绑定工具）
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点
        tool_node = ToolNode(all_tools)

        # 构建消息
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具
3. 调用工具获取信息
4. 返回执行结果

注意：不要编造数据，只返回实际获取的信息"""),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # 没有工具调用，直接使用 LLM 的输出
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        return {
            "plan": plan[1:],  # 移除已执行的步骤
            "past_steps": [(task, result)],  # 添加到执行历史
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}")
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }
```

### 调用关系
- **被谁调用**：aiops_service.py (LangGraph 工作流)
- **调用谁**：ChatQwen, ToolNode, tools, mcp_client

### 初学者最容易误解的地方
1. **ToolNode 的作用**：自动执行 LLM 返回的工具调用
2. **bind_tools()**：将工具绑定到 LLM，让 LLM 知道有哪些工具可用
3. **plan[1:]**：切片操作，移除第一个元素（已执行的步骤）
4. **past_steps 追加**：由于 state 定义中使用了 `operator.add`，这里返回的 past_steps 会自动追加到现有列表中

---

## 7. app/agent/aiops/replanner.py - Replanner 节点

### 文件作用
评估执行结果，决定是继续执行、调整计划还是生成最终响应。

### 手敲代码

```python
# app/agent/aiops/replanner.py
from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from app.agent.aiops.state import PlanExecuteState
from app.agent.aiops.utils import format_tools_description


class Response(BaseModel):
    """最终响应的格式"""
    response: str = Field(description="对用户的最终响应")


class Act(BaseModel):
    """重新规划的输出格式"""
    action: str = Field(
        description="""下一步的行动：
        - 'continue': 继续执行下一个步骤
        - 'replan': 调整计划
        - 'respond': 生成最终响应"""
    )
    new_steps: List[str] = Field(
        default_factory=list,
        description="新的步骤列表（如果 action 是 'replan'）"
    )


replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

        你有三个选择：
        1. 'respond' - 信息充足，立即生成最终响应 【最高优先级】
        2. 'continue' - 继续执行当前计划
        3. 'replan' - 调整计划 【谨慎使用】

        评估标准：
        - 当前信息是否已经足够解决用户问题？
        - 已执行步骤是否成功获取了核心信息？
    """).strip()),
    ("placeholder", "{messages}"),
])

response_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

        响应要求：
        - 清晰、结构化
        - 基于实际数据，不要编造
        - 使用 Markdown 格式
    """).strip()),
    ("placeholder", "{messages}"),
])


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """重新规划节点：决定下一步行动"""
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    # 强制限制：如果已执行步骤过多，直接生成响应
    MAX_STEPS = 8
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"已执行 {len(past_steps)} 个步骤，强制生成最终响应")
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        return await _generate_response(state, llm)

    # 创建 LLM
    llm = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0
    )

    # 格式化已执行的步骤
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:300]}..."
        for step, result in past_steps
    ])

    # 如果还有剩余计划，进行决策
    if plan:
        logger.info("还有剩余计划，评估下一步行动")

        replanner_chain = replanner_prompt | llm.with_structured_output(Act)

        try:
            messages = [
                ("user", f"原始任务: {input_text}"),
                ("user", f"已执行的步骤:\n{steps_summary}"),
                ("user", f"剩余计划: {', '.join(plan)}")
            ]

            act = await replanner_chain.ainvoke({
                "messages": messages,
                "tools_description": format_tools_description(
                    [get_current_time, retrieve_knowledge]
                )
            })

            if isinstance(act, Act):
                action = act.action
                new_steps = act.new_steps
            else:
                action = act.get("action", "continue")
                new_steps = act.get("new_steps", [])

            logger.info(f"Replanner 决策: {action}")

            if action == "respond":
                logger.info("决定生成最终响应")
                return await _generate_response(state, llm)

            elif action == "replan":
                if len(new_steps) > len(plan):
                    new_steps = new_steps[:len(plan)]

                if len(past_steps) >= 5:
                    logger.warning(f"已执行 {len(past_steps)} 个步骤，禁止重新规划")
                    return await _generate_response(state, llm)

                logger.info(f"决定调整计划，新步骤数量: {len(new_steps)}")
                if new_steps:
                    return {"plan": new_steps}
                else:
                    return {}

            else:  # action == "continue"
                logger.info("决定继续执行当前计划")
                return {}

        except Exception as e:
            logger.error(f"重新规划失败: {e}")
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state, llm)


async def _generate_response(state: PlanExecuteState, llm: ChatQwen) -> Dict[str, Any]:
    """生成最终响应"""
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])

    response_gen = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务: {input_text}"),
            ("user", f"执行历史:\n{execution_history}"),
            ("user", "请基于以上信息生成全面的最终响应")
        ]

        response_obj = await response_gen.ainvoke({"messages": messages})

        if isinstance(response_obj, Response):
            final_response = response_obj.response
        else:
            final_response = response_obj.get("response", "")

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")

        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        fallback_response = f"# 任务执行结果\n\n原始任务: {input_text}\n\n已执行 {len(past_steps)} 个步骤"
        return {"response": fallback_response}
```

### 调用关系
- **被谁调用**：aiops_service.py (LangGraph 工作流)
- **调用谁**：ChatQwen, tools, mcp_client

### 初学者最容易误解的地方
1. **三种决策**：respond（最高优先级）、continue（次优先级）、replan（谨慎使用）
2. **MAX_STEPS 限制**：防止无限循环，超过 8 步强制生成响应
3. **action 优先级**：优先检查 respond，然后是 replan，最后是 continue

---

## 8. app/services/aiops_service.py - AIOps 服务

### 文件作用
构建 LangGraph 工作流，协调 Planner-Executor-Replanner 三个节点。

### 手敲代码

```python
# app/services/aiops_service.py
from typing import AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.aiops import PlanExecuteState, planner, executor, replanner


# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service 初始化完成")

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        # 创建状态图
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_REPLANNER, replanner)

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 定义边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        # replanner 的条件边
        def should_continue(state: PlanExecuteState) -> str:
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END
            }
        )

        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """执行 Plan-Execute-Replan 流程"""
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": ""
            }

            config_dict = {"configurable": {"thread_id": session_id}}

            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)
                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)
                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成"
            }

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}")
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """AIOps 诊断接口"""
        # 使用固定的 AIOps 任务描述
        from textwrap import dedent
        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告。""")

        async for event in self.execute(aiops_task, session_id):
            if event.get("type") == "complete":
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {"status": "completed"}
                }
            else:
                yield event

    def _format_planner_event(self, state: Dict | None) -> Dict:
        plan = state.get("plan", []) if state else []
        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan
        }

    def _format_executor_event(self, state: Dict | None) -> Dict:
        if not state:
            return {"type": "status", "stage": "executor", "message": "执行节点运行中"}

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan)
            }
        return {"type": "status", "stage": "executor", "message": "开始执行步骤"}

    def _format_replanner_event(self, state: Dict | None) -> Dict:
        response = state.get("response", "") if state else ""
        plan = state.get("plan", []) if state else []

        if response:
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response
            }
        return {
            "type": "status",
            "stage": "replanner",
            "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}"
        }


# 全局单例
aiops_service = AIOpsService()
```

### 调用关系
- **被谁调用**：api/aiops.py
- **调用谁**：planner, executor, replanner

### 初学者最容易误解的地方
1. **StateGraph 的作用**：定义工作流的状态和节点之间的关系
2. **条件边**：`add_conditional_edges` 允许根据状态决定下一个节点
3. **stream_mode="updates"**：只返回节点的输出，而不是整个状态

---

## 9. app/api/aiops.py - AIOps 接口

### 手敲代码

```python
# app/api/aiops.py
import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.aiops import AIOpsRequest
from app.services.aiops_service import aiops_service

router = APIRouter()


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    """AIOps 故障诊断接口（流式 SSE）"""
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求")

    async def event_generator():
        try:
            async for event in aiops_service.diagnose(session_id=session_id):
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                if event.get("type") in ["complete", "error"]:
                    break

        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断异常: {e}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"诊断异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())
```

---

## 10. app/api/file.py - 文件上传接口

### 手敲代码

```python
# app/api/file.py
import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.vector_store_manager import vector_store_manager
from app.services.document_splitter_service import document_splitter_service

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并添加到知识库"""
    try:
        logger.info(f"收到文件上传请求: {file.filename}")

        # 读取文件内容
        content = await file.read()
        text = content.decode("utf-8")

        # 删除旧数据
        file_path = file.filename or "unknown"
        vector_store_manager.delete_by_source(file_path)

        # 分割文档
        chunks = document_splitter_service.split_text(
            text,
            file_name=file.filename
        )

        # 添加到向量库
        vector_store_manager.add_documents(chunks)

        logger.info(f"文件处理完成: {file.filename}, 共 {len(chunks)} 个分块")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "file_name": file.filename,
                "chunk_count": len(chunks)
            }
        }

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": "error",
                "data": {"error": str(e)}
            }
        )
```

---

## 11. 修改 app/main.py - 添加路由和静态文件

```python
# 在 main.py 中添加以下代码

# ... (已有的代码)

# 注册路由
from app.api import health, chat, file, aiops

app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
from fastapi.staticfiles import StaticFiles
import os

static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }
```

---

## 阶段三检查清单

完成后，你应该能够：

- [ ] 使用 `/api/aiops` 发起智能诊断
- [ ] 看到 Plan-Execute-Replan 工作流正常执行
- [ ] 使用 `/api/upload` 上传文档到知识库
- [ ] 访问 Web 界面 `http://localhost:9900`
- [ ] AIOps 诊断能正确调用 MCP 工具

### 测试命令

```bash
# AIOps 诊断（流式）
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-aiops"}' \
  --no-buffer
```

---

## 完整项目检查清单

### 阶段一
- [ ] config.py - 配置管理
- [ ] utils/logger.py - 日志配置
- [ ] main.py - FastAPI 入口

### 阶段二
- [ ] models/request.py - 请求模型
- [ ] models/response.py - 响应模型
- [ ] core/milvus_client.py - Milvus 客户端
- [ ] services/vector_store_manager.py - 向量存储管理
- [ ] services/vector_embedding_service.py - 向量嵌入服务
- [ ] tools/knowledge_tool.py - 知识检索工具
- [ ] tools/time_tool.py - 时间工具
- [ ] agent/mcp_client.py - MCP 客户端
- [ ] services/rag_agent_service.py - RAG Agent 服务
- [ ] api/health.py - 健康检查接口
- [ ] api/chat.py - 对话接口

### 阶段三
- [ ] models/aiops.py - AIOps 请求模型
- [ ] agent/aiops/state.py - 状态定义
- [ ] agent/aiops/utils.py - 工具函数
- [ ] agent/aiops/__init__.py - 模块导出
- [ ] agent/aiops/planner.py - Planner 节点
- [ ] agent/aiops/executor.py - Executor 节点
- [ ] agent/aiops/replanner.py - Replanner 节点
- [ ] services/aiops_service.py - AIOps 服务
- [ ] api/aiops.py - AIOps 接口
- [ ] api/file.py - 文件上传接口
- [ ] main.py 更新 - 路由和静态文件

---

## 总结

恭喜你完成了整个项目的手敲！

### 你已经掌握的核心概念

1. **FastAPI 框架**
   - 路由定义和中间件配置
   - 请求/响应模型（Pydantic）
   - SSE 流式输出

2. **LangChain / LangGraph**
   - Agent 创建和工具绑定
   - StateGraph 工作流定义
   - 状态管理和检查点

3. **RAG 模式**
   - 向量存储和检索
   - 文档分割和向量化
   - 检索增强生成

4. **AIOps 智能诊断**
   - Plan-Execute-Replan 模式
   - 多节点协作
   - 动态决策和重规划

### 下一步建议

1. 添加更多工具（如数据库查询、API 调用）
2. 优化提示词提高诊断准确性
3. 添加前端界面
4. 实现用户认证和权限管理
5. 部署到生产环境
