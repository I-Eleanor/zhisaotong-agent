"""诊断 Agent 数据结构（Pydantic）。

tool 必须限制在允许的工具白名单内（见 ALLOWED_TOOLS），
由解析层（parser.py）与执行层（tool_router.py）共同保证。

字段别名：LLM 输出中 "steps" 与 "plan" 两种键名均接受（AliasChoices），
兼容新旧提示词与不同模型的输出习惯。
"""
from typing import Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

# 工具白名单：诊断步骤只允许调用这些工具
ALLOWED_TOOLS: tuple[str, ...] = (
    "query_device_status",
    "query_error_code",
    "query_maintenance",
    "retrieve_knowledge",
)

# 计划最多步数 / 最大迭代轮次（防无限循环）
MAX_STEPS = 5
MAX_ITERATIONS = 5


class DiagnosticStep(BaseModel):
    """单个排查步骤。"""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    description: str
    tool: str
    arguments: dict[str, str] = Field(default_factory=dict)

    @field_validator("tool")
    @classmethod
    def _validate_tool(cls, value: str) -> str:
        value = (value or "").strip()
        if value not in ALLOWED_TOOLS:
            raise ValueError(f"非法工具名：{value}，允许的工具：{ALLOWED_TOOLS}")
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("步骤描述不能为空")
        return value.strip()

    @field_validator("arguments", mode="before")
    @classmethod
    def _stringify_arguments(cls, value: object) -> object:
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return value


class DiagnosticPlan(BaseModel):
    """排查计划。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    steps: list[DiagnosticStep] = Field(
        default_factory=list,
        validation_alias=AliasChoices("steps", "plan"),
    )


class StepResult(BaseModel):
    """步骤执行结果。

    失败时携带 error_code 与 safe_error_message（面向用户的安全信息），
    内部异常细节只进日志，不得伪装成正常诊断结果。
    """

    success: bool
    content: str = ""
    error_code: str = ""
    safe_error_message: str = ""


class CompletedStep(BaseModel):
    """已完成的步骤（步骤 + 执行结果）。"""

    step: DiagnosticStep
    result: StepResult


class ReplanDecision(BaseModel):
    """重规划决策：continue / replan / end。

    replan 时 steps 表示新的待执行步骤（不含已完成步骤，从第一项开始执行）。
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    action: Literal["continue", "replan", "end"] = "continue"
    reason: str = ""
    steps: list[DiagnosticStep] | None = Field(
        default=None,
        validation_alias=AliasChoices("steps", "plan"),
    )

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value
