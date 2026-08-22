from collections.abc import Callable
from typing import Any, cast

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import log_safe_text, log_safe_value, logger, safe_exception_fields
from utils.prompt_loader import load_report_prompts, load_system_prompts


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info({
        "event": "tool_monitor_start",
        "tool": log_safe_text(request.tool_call["name"]),
        "args": log_safe_value(request.tool_call["args"]),
    })

    try:
        result = handler(request)
        logger.info({"event": "tool_monitor_success", "tool": log_safe_text(request.tool_call["name"])})

        if request.tool_call['name'] == "fill_context_for_report":
            # 运行时 context 实际是 dict（create_agent 默认），静态类型推断为 None，这里显式收窄
            cast(dict[str, Any], request.runtime.context)["report"] = True

        return result
    except Exception as e:
        logger.error({
            "event": "tool_monitor_error",
            "tool": log_safe_text(request.tool_call["name"]),
            **safe_exception_fields(e),
        })
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info({
        "event": "before_model_call",
        "message_count": len(state['messages']),
    })

    logger.debug({
        "event": "before_model_last_message",
        "message_type": type(state['messages'][-1]).__name__,
        "content": log_safe_text(str(state['messages'][-1].content)),
    })

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = cast(dict[str, Any], request.runtime.context).get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
