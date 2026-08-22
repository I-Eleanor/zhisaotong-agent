"""工具路由层：白名单、参数校验与补全、调用与错误包装。

- 只有 ALLOWED_TOOLS 内的工具会被执行，其余返回 TOOL_UNAVAILABLE；
- 必填参数缺失时尝试从上下文补全（如 user_id 自动取当前用户、query 兜底为用户原始问题）；
- 调用异常一律转为 StepResult(success=False)，携带 error_code 与 safe_error_message，
  原始异常只进日志，不得伪装成正常诊断结果。
"""
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from agent.diagnostic.schemas import ALLOWED_TOOLS, DiagnosticStep, StepResult
from agent.tools.diagnostic_tools import current_user_id
from utils import error_codes
from utils.exceptions import ServiceUnavailableError
from utils.logger_handler import log_safe_value, logger, safe_exception_fields

RawTool = Callable[..., str]

_ERROR_CODE_PATTERN = re.compile(r"\bE\d{1,3}\b", re.IGNORECASE)


def _fill_user_id(user_query: str) -> str | None:
    return current_user_id()


def _fill_error_code(user_query: str) -> str | None:
    match = _ERROR_CODE_PATTERN.search(user_query or "")
    return match.group(0).upper() if match else None


def _fill_query(user_query: str) -> str | None:
    return user_query or None


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的调用规格：函数、必填参数、缺失参数补全规则。"""

    func: RawTool
    required: tuple[str, ...]
    auto_fill: dict[str, Callable[[str], str | None]] = field(default_factory=dict)


def build_default_tool_specs(knowledge_agent=None) -> dict[str, ToolSpec]:
    """构造默认工具规格；knowledge_agent 可注入（应用容器管理的知识库 Agent），
    未注入时知识类工具回退全局懒加载单例。"""
    from agent.tools.diagnostic_tools import (
        build_raw_knowledge_tools,
        raw_query_device_status,
    )

    raw_error_code, raw_maintenance, raw_knowledge = build_raw_knowledge_tools(knowledge_agent)

    return {
        "query_device_status": ToolSpec(
            func=raw_query_device_status,
            required=("user_id",),
            auto_fill={"user_id": _fill_user_id},
        ),
        "query_error_code": ToolSpec(
            func=raw_error_code,
            required=("error_code",),
            auto_fill={"error_code": _fill_error_code},
        ),
        "query_maintenance": ToolSpec(
            func=raw_maintenance,
            required=("query",),
            auto_fill={"query": _fill_query},
        ),
        "retrieve_knowledge": ToolSpec(
            func=raw_knowledge,
            required=("query",),
            auto_fill={"query": _fill_query},
        ),
    }


class ToolRouter:
    """按步骤路由到白名单工具并包装执行结果。

    specs / knowledge_agent 均可注入（测试替身/容器管理的依赖），
    默认使用 diagnostic_tools 的 raw 函数与全局知识库 Agent。
    """

    def __init__(self, specs: dict[str, ToolSpec] | None = None, knowledge_agent=None):
        if specs is not None:
            self._specs: dict[str, ToolSpec] = specs
        else:
            self._specs = build_default_tool_specs(knowledge_agent=knowledge_agent)

    def execute(self, step: DiagnosticStep, user_query: str = "") -> StepResult:
        """执行单个步骤并返回结构化结果；任何失败都不携带原始异常文本。"""
        spec = self._specs.get(step.tool)
        if spec is None:
            logger.warning({"event": "tool_router_unavailable", "tool": step.tool})
            return StepResult(
                success=False,
                error_code=error_codes.TOOL_UNAVAILABLE,
                safe_error_message=f"诊断工具 {step.tool} 暂不可用，本步骤已跳过。",
            )

        args = self._resolve_arguments(step, spec, user_query)
        missing = [name for name in spec.required if not args.get(name)]
        if missing:
            logger.warning({
                "event": "tool_router_missing_args",
                "tool": step.tool,
                "missing": missing,
            })
            return StepResult(
                success=False,
                error_code=error_codes.TOOL_ARGUMENT_INVALID,
                safe_error_message=f"步骤「{step.description}」缺少必要参数（{','.join(missing)}），本步骤已跳过。",
            )

        try:
            content = spec.func(**args)
        except ServiceUnavailableError as e:
            # 底层服务明确报告不可用（项目异常）：转为失败步骤，绝不把
            # 错误字符串当成成功结果写入已确认事实。
            logger.warning({
                "event": "tool_router_service_unavailable",
                "tool": step.tool,
                "args": log_safe_value(args),
                "error_code": e.error_code,
                **safe_exception_fields(e),
            })
            return StepResult(
                success=False,
                error_code=e.error_code,
                safe_error_message=f"步骤「{step.description}」的底层服务暂时不可用，本步骤已跳过。",
            )
        except Exception as e:
            logger.error({
                "event": "tool_router_execution_error",
                "tool": step.tool,
                "args": log_safe_value(args),
                **safe_exception_fields(e),
            })
            return StepResult(
                success=False,
                error_code=error_codes.TOOL_EXECUTION_FAILED,
                safe_error_message=f"步骤「{step.description}」的工具调用失败，本步骤已跳过。",
            )

        logger.info({
            "event": "tool_router_success",
            "tool": step.tool,
            "args": log_safe_value(args),
            "result_length": len(content or ""),
        })
        return StepResult(success=True, content=str(content))

    def _resolve_arguments(self, step: DiagnosticStep, spec: ToolSpec, user_query: str) -> dict[str, str]:
        """取交集参数并按补全规则填充缺失的必填参数。"""
        args: dict[str, str] = {}
        for name, value in step.arguments.items():
            if name in spec.required and value:
                args[name] = str(value)
            elif name not in spec.required:
                logger.warning({
                    "event": "tool_router_unknown_arg_dropped",
                    "tool": step.tool,
                    "arg": name,
                })
        for name in spec.required:
            if args.get(name):
                continue
            filler = spec.auto_fill.get(name)
            if filler is not None:
                filled = filler(user_query)
                if filled:
                    args[name] = filled
        return args


def allowed_tools() -> tuple[str, ...]:
    """返回工具白名单（供提示词与校验复用）。"""
    return ALLOWED_TOOLS
