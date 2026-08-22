"""集中解析层（LLM structured output 兼容层）。

所有「LLM 输出 → Pydantic 模型」的解析统一经由本模块，节点里不得散落正则解析：
1. 优先尝试模型的 with_structured_output()（structured output 能力）；
2. 模型或测试替身不支持时，降级为「自由文本 → JSON 提取 → 归一化 → Pydantic 校验」；
3. 任一环节失败返回 None，由调用方决定兜底策略（固定计划 / 按原计划继续 / 强制结束）。
"""
import json
import re
from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from agent.diagnostic.schemas import DiagnosticPlan, DiagnosticStep, ReplanDecision
from model.factory import get_chat_model
from utils.logger_handler import logger, safe_exception_fields

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def extract_json(text: str):
    """从 LLM 输出中尽力解析出 JSON（兼容 ```json 代码块与裸 JSON）。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退而求其次：截取第一个 [ 或 { 到最后一个 ] 或 }
    arr = re.search(r"\[.*\]", text, re.DOTALL)
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    for match in (arr, obj):
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None


def _coerce_step_item(item: object) -> object:
    """单个步骤变体归一化：纯字符串步骤降级为全库检索步骤。"""
    if isinstance(item, str):
        return {"description": item, "tool": "retrieve_knowledge", "arguments": {}}
    return item


def _sanitize_step_items(items: list) -> list:
    """逐项校验步骤：非法项（如非法工具名、缺描述）只丢弃自身，合法步骤保留。

    日志只记丢弃数量，不记录步骤内容（避免 LLM 输出 / 用户输入泄漏）。
    """
    valid = []
    for item in items:
        try:
            valid.append(DiagnosticStep.model_validate(item).model_dump())
        except ValidationError:
            logger.warning({"event": "llm_parse_step_dropped"})
    return valid


def _coerce_plan_data(data: object) -> object:
    """计划变体归一化：裸数组 / {"plan": [...]} / {"steps": [...]} 统一为 {"steps": [...]}。"""
    if isinstance(data, list):
        data = {"steps": data}
    if isinstance(data, dict):
        steps = data.get("steps", data.get("plan"))
        if isinstance(steps, list):
            items = [_coerce_step_item(i) for i in steps]
            return {"steps": _sanitize_step_items(items)}
    return data


def _coerce_replan_data(data: object) -> object:
    """重规划决策变体归一化：steps/plan 别名统一，步骤逐项校验后保留合法项。"""
    if isinstance(data, dict):
        steps = data.get("steps", data.get("plan"))
        if isinstance(steps, list):
            items = [_coerce_step_item(i) for i in steps]
            rest = {k: v for k, v in data.items() if k not in ("steps", "plan")}
            return {**rest, "steps": _sanitize_step_items(items)}
    return data


class LlmParser:
    """结构化输出解析器，model 可注入（测试替身/不同后端）。"""

    def __init__(self, model=None):
        self._model = model

    def _get_model(self):
        return self._model if self._model is not None else get_chat_model()

    @staticmethod
    def _messages(system_prompt: str, user_prompt: str) -> list[BaseMessage]:
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    def parse_plan(self, system_prompt: str, user_prompt: str) -> DiagnosticPlan | None:
        """解析排查计划；失败返回 None（由 planner 兜底为固定计划）。"""
        parsed = self._parse(
            DiagnosticPlan,
            system_prompt,
            user_prompt,
            coerce=_coerce_plan_data,
        )
        return parsed if isinstance(parsed, DiagnosticPlan) else None

    def parse_replan(self, system_prompt: str, user_prompt: str) -> ReplanDecision | None:
        """解析重规划决策；失败返回 None（由 replanner 决定继续或结束）。"""
        parsed = self._parse(
            ReplanDecision,
            system_prompt,
            user_prompt,
            coerce=_coerce_replan_data,
        )
        return parsed if isinstance(parsed, ReplanDecision) else None

    def _parse(
        self,
        schema: type[BaseModelT],
        system_prompt: str,
        user_prompt: str,
        coerce=None,
    ) -> BaseModelT | None:
        messages = self._messages(system_prompt, user_prompt)
        # 1) structured output 优先
        obj = self._invoke_structured(schema, messages)
        if obj is not None:
            return obj
        # 2) 降级：自由文本 → JSON 提取 → 归一化 → Pydantic 校验
        text = self._invoke_text(messages)
        if text is None:
            return None
        data = extract_json(text)
        if data is None:
            logger.warning({"event": "llm_parse_no_json", "schema": schema.__name__})
            return None
        if coerce is not None:
            data = coerce(data)
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            # include_input=False：errors() 默认携带 "input" 字段，
            # 会把 LLM 原始输出（含用户输入）写进日志，必须剔除。
            logger.warning({
                "event": "llm_parse_validation_failed",
                "schema": schema.__name__,
                "errors": e.errors(include_url=False, include_input=False)[:3],
            })
            return None

    def _invoke_structured(self, schema: type[BaseModelT], messages: list[BaseMessage]) -> BaseModelT | None:
        """尝试 structured output；模型/替身不支持或调用失败时返回 None 走降级。"""
        model = self._get_model()
        try:
            runner = model.with_structured_output(schema)
        except Exception as e:
            logger.info({
                "event": "structured_output_unsupported",
                "error_type": type(e).__name__,
            })
            return None
        try:
            obj = runner.invoke(messages)
        except Exception as e:
            logger.warning({
                "event": "structured_output_failed",
                **safe_exception_fields(e),
            })
            return None
        if isinstance(obj, schema):
            return obj
        return None

    def _invoke_text(self, messages: list[BaseMessage]) -> str | None:
        """普通调用并返回文本内容；调用失败返回 None。"""
        model = self._get_model()
        try:
            resp = model.invoke(messages)
        except Exception as e:
            logger.error({
                "event": "llm_invoke_error",
                **safe_exception_fields(e),
            })
            return None
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return content
