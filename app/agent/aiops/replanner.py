"""
Replanner 节点：重新规划或生成最终响应
基于 LangGraph 官方教程实现
"""

from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState
from .utils import format_tools_description


class Response(BaseModel):
    """最终响应的格式"""
    response: str = Field(description="对用户的最终响应")


class Act(BaseModel):
    """重新规划的输出格式"""
    action: str = Field(
        description="""下一步的行动，必须是以下三种之一：
        - 'continue': 当前计划合理，继续执行下一个步骤
        - 'replan': 当前计划需要调整，提供新的步骤列表
        - 'respond': 计划已完成且信息充足，生成最终响应"""
    )
    # action 为 'replan' 时，新的步骤列表（会替换当前剩余计划）
    new_steps: List[str] = Field(
        default_factory=list,
        description="新的步骤列表（如果 action 是 'replan'，这些步骤会替换剩余计划）"
    )


# Replanner 提示词
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                你有三个选择（按优先级排序）：

                **1. 'respond' - 信息充足，立即生成最终响应** 【最高优先级】
                   - 使用场景：当前信息已经足够回答用户问题
                   - 决策标准：
                     * 已执行步骤 >= 3 且获取了关键信息
                     * 或者已执行步骤 >= 5（无论结果如何）
                     * 或者当前信息完全满足任务需求
                   - ⚠️ 不要等到"完美"才响应，"足够好"就应该立即 respond

                **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
                   - 使用场景：剩余计划合理且必要
                   - 决策标准：剩余步骤确实能提供关键信息
                   - ⚠️ 如果剩余步骤不是"必需"的，应选择 respond

                **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】
                   - 使用场景：原计划明显错误或遗漏关键步骤
                   - ⚠️ **严格限制**：
                     * 新步骤数量必须 <= 当前剩余步骤数
                     * 优先简化计划，不要添加不必要的步骤
                     * 总步骤数已执行 >= 5 次时，禁止 replan，只能 respond

                评估标准：
                - 当前信息是否已经足够解决用户问题？【最关键】
                - 已执行步骤是否成功获取了核心信息？
                - 剩余步骤是否真的"必需"？
                - 已执行步骤数是否过多（>= 5）？如果是，立即 respond

                **决策优先级口诀：** 
                "优先结束 > 保持不变 > 调整计划"
                "信息足够就响应，不要追求完美"
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应

    三种决策：
    1. continue - 继续执行当前计划
    2. replan - 调整计划（替换剩余步骤）
    3. respond - 生成最终响应
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {len(past_steps)}")

    # ⚠️ 强制限制：如果已执行步骤过多，直接生成响应
    MAX_STEPS = 8
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"已执行 {len(past_steps)} 个步骤，超过最大限制 {MAX_STEPS}，强制生成最终响应")
        return await _generate_response(state)

    # 获取可用工具列表
    try:
        # 获取本地工具
        local_tools = [
            get_current_time,
            retrieve_knowledge
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)
    except Exception as e:
        logger.warning(f"获取工具列表失败: {e}")
        tools_description = "无法获取工具列表"

    # 创建 LLM
    llm = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=120.0  # 设置 120 秒超时
    )

    # 格式化已执行的步骤（安全处理非字符串结果）
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {str(result)[:300]}..."
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
                ("user", f"剩余计划: {', '.join(plan)}"),
                ("user", f"⚠️ 重要提示：已执行 {len(past_steps)} 个步骤，请优先考虑是否信息已足够生成响应（respond）")
            ]

            act = await replanner_chain.ainvoke({
                "messages": messages,
                "tools_description": tools_description
            })

            # 处理返回结果
            if isinstance(act, Act):
                action = act.action
                new_steps = act.new_steps
            else:
                # 如果返回的是字典
                action = act.get("action", "continue")  # type: ignore
                new_steps = act.get("new_steps", [])  # type: ignore

            logger.info(f"Replanner 决策: {action}")

            if action == "respond":
                logger.info("决定生成最终响应")
                return await _generate_response(state)

            elif action == "replan":
                # ⚠️ 强制限制：新步骤数不能超过当前剩余步骤数
                if len(new_steps) > len(plan):
                    logger.warning(
                        f"新步骤数 {len(new_steps)} > 剩余步骤数 {len(plan)}，"
                        f"强制截断为 {len(plan)} 个步骤"
                    )
                    new_steps = new_steps[:len(plan)]
                
                # ⚠️ 二次检查：如果已执行步骤 >= 5，禁止 replan
                if len(past_steps) >= 5:
                    logger.warning(f"已执行 {len(past_steps)} 个步骤，禁止重新规划，强制生成响应")
                    return await _generate_response(state)
                
                logger.info(f"决定调整计划，新步骤数量: {len(new_steps)}")
                if new_steps:
                    # 替换剩余计划
                    return {"plan": new_steps}
                else:
                    logger.warning("replan 但未提供新步骤，继续执行原计划")
                    return {}

            else:  # action == "continue"
                logger.info("决定继续执行当前计划")
                return {}  # 不修改状态，继续执行

        except Exception as e:
            logger.error(f"重新规划失败: {e}, 继续执行剩余计划")
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state)


def _summarize_step_result(result: str, max_chars: int = 500) -> str:
    """摘要化工具执行结果，避免将大量原始数据传给 LLM。

    对于包含 JSON 数据的步骤，提取关键指标和统计信息，
    而不是发送完整的数据点列表。
    """
    if len(result) <= max_chars:
        return result

    # 尝试提取关键信息：统计信息、告警信息等
    import json

    # 先尝试从文本中提取 JSON 对象
    summary_parts = []
    try:
        # 查找第一个完整的 JSON 对象
        json_start = result.index('{')
        json_end = result.rindex('}') + 1
        json_str = result[json_start:json_end]
        data = json.loads(json_str)

        # 提取统计信息
        if 'statistics' in data:
            stats = data['statistics']
            stat_items = []
            for key, val in stats.items():
                if isinstance(val, (int, float, bool, str)):
                    stat_items.append(f"  - {key}: {val}")
            if stat_items:
                summary_parts.append("统计信息:\n" + "\n".join(stat_items))

        # 提取告警信息
        if 'alert_info' in data:
            alert = data['alert_info']
            alert_items = []
            for key, val in alert.items():
                if isinstance(val, (int, float, bool, str)):
                    alert_items.append(f"  - {key}: {val}")
            if alert_items:
                summary_parts.append("告警信息:\n" + "\n".join(alert_items))

        # 提取关键元数据（排除 data_points）
        for key in ('service_name', 'metric_name', 'interval'):
            if key in data:
                summary_parts.append(f"{key}: {data[key]}")

        # 如果有 data_points，只取摘要
        if 'data_points' in data:
            points = data['data_points']
            if isinstance(points, list) and points:
                summary_parts.append(
                    f"数据点数量: {len(points)} 个\n"
                    f"时间范围: {points[0].get('timestamp', '?')} 至 {points[-1].get('timestamp', '?')}"
                )

        if summary_parts:
            return "\n".join(summary_parts)

    except (ValueError, json.JSONDecodeError, KeyError):
        pass  # 不是 JSON 格式，走默认截断

    # 默认：截取前 max_chars 字符
    return result[:max_chars] + f"\n... (结果过长，已截断，总长度 {len(result)} 字符)"


async def _generate_response(state: PlanExecuteState) -> Dict[str, Any]:
    """生成最终响应

    使用较长的超时时间（300s），并对执行历史进行摘要化，
    避免大量原始监控数据导致请求超时。
    """
    logger.info("生成最终响应...")

    # 为最终响应生成创建独立的 LLM，使用更长超时时间
    llm = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=300.0  # 最终响应生成需要更长时间
    )

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    # 摘要化执行历史，避免将大量原始监控数据传给 LLM 导致超时
    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果摘要:**\n{_summarize_step_result(str(result))}"
        for step, result in past_steps
    ])
    logger.info(f"执行历史摘要长度: {len(execution_history)} 字符")

    response_gen = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务: {input_text}"),
            ("user", f"执行历史:\n{execution_history}"),
            ("user", "请基于以上信息生成全面的最终响应")
        ]

        response_obj = await response_gen.ainvoke({"messages": messages})

        # 处理返回结果
        if isinstance(response_obj, Response):
            final_response = response_obj.response
        else:
            # 如果返回的是字典
            final_response = response_obj.get("response", "")  # type: ignore

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")

        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        # 生成简单的后备响应
        fallback_response = f"""# 任务执行结果

## 原始任务
{input_text}

## 执行的步骤
{_format_simple_steps(past_steps)}

## 说明
由于系统异常，无法生成完整响应。以上是已收集的信息。
"""
        return {"response": fallback_response}


def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表（简单版）"""
    if not past_steps:
        return "无"

    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        result_preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {result_preview}\n")

    return "\n".join(formatted)
