"""诊断 Agent 重构测试（P0-2）。

覆盖场景（对应修改指令三.11）：
- replan 后执行新计划第一步（不跳过、不重复）
- 新计划缩短 / 新计划为空
- 非法工具名（schema 校验 + 路由层兜底）
- 工具调用失败（StepResult 携带 error_code / safe_error_message）
- 达到最大迭代次数强制结束
- LLM 返回非法结构（解析层兜底）
- 已完成步骤不会重复执行
- 最终一定产生 report 和 done 事件
- 解析层：structured output 优先、别名兼容、字符串步骤降级
"""
import json

import pytest
from pydantic import ValidationError

from agent.diagnostic.parser import LlmParser
from agent.diagnostic.schemas import (
    ALLOWED_TOOLS,
    MAX_ITERATIONS,
    MAX_STEPS,
    CompletedStep,
    DiagnosticPlan,
    DiagnosticStep,
    ReplanDecision,
    StepResult,
)
from agent.diagnostic.tool_router import ToolRouter, ToolSpec


# ----------------------------------------------------------------- 测试替身
class StubParser:
    """按脚本返回计划/决策的解析替身；decisions 按调用次序弹出。"""

    def __init__(self, plan=None, decisions=None):
        self.plan = plan
        self.decisions = list(decisions or [])
        self.plan_calls = 0
        self.replan_calls = 0

    def parse_plan(self, system_prompt, user_prompt):
        self.plan_calls += 1
        return self.plan

    def parse_replan(self, system_prompt, user_prompt):
        self.replan_calls += 1
        if self.replan_calls <= len(self.decisions):
            return self.decisions[self.replan_calls - 1]
        return ReplanDecision(action="end", reason="测试默认结束")


class RecordingRouter:
    """记录执行顺序、按脚本返回结果的工具路由替身。"""

    def __init__(self, results=None, fail_tools=()):
        self.executed: list[DiagnosticStep] = []
        self.results = list(results or [])
        self.fail_tools = set(fail_tools)

    def execute(self, step, user_query=""):
        self.executed.append(step)
        if step.tool in self.fail_tools:
            return StepResult(
                success=False,
                error_code="TOOL_EXECUTION_FAILED",
                safe_error_message=f"步骤「{step.description}」的工具调用失败，本步骤已跳过。",
            )
        if self.results:
            return self.results.pop(0)
        return StepResult(success=True, content=f"{step.description}的结果")


class StubReportModel:
    """reporter 用的假模型：返回固定报告。"""

    def invoke(self, messages, **kw):
        class R:
            content = "## 诊断报告\n### 已确认事实\n测试事实"
        return R()


def _step(desc, tool="query_device_status", args=None):
    return DiagnosticStep(description=desc, tool=tool, arguments=args or {})


def _run_agent(parser, router, query="扫地机不工作"):
    from agent.diagnostic.service import DiagnosticAgent

    agent = DiagnosticAgent(parser=parser, tool_router=router, model=StubReportModel())
    return list(agent.run(query))


def _events_of_type(events, type_):
    return [e for e in events if e["type"] == type_]


# ----------------------------------------------------------------- 状态机核心场景
def test_replan_executes_new_plan_first_step():
    """replan 后：新计划第一项被执行，旧 pending 被整体替换（不跳过、不重复）。"""
    plan = DiagnosticPlan(steps=[_step("步骤A1"), _step("步骤A2")])
    new_step = _step("新步骤C1", tool="retrieve_knowledge")
    parser = StubParser(
        plan=plan,
        decisions=[ReplanDecision(action="replan", reason="换个方向", steps=[new_step])],
    )
    router = RecordingRouter()

    events = _run_agent(parser, router)

    assert [s.description for s in router.executed] == ["步骤A1", "新步骤C1"], (
        "应先执行旧计划第一步，replan 后从新计划第一项开始执行"
    )
    replan_ev = _events_of_type(events, "replan")
    assert any("新步骤C1" in e["content"] for e in replan_ev)
    assert events[-1]["type"] == "done"
    assert _events_of_type(events, "report"), "最终必须产生 report"


def test_replan_shorter_plan_still_ends_cleanly():
    """新计划缩短：3 步计划执行 1 步后 replan 为 1 步，总执行 2 步后正常结束。"""
    plan = DiagnosticPlan(steps=[_step("A1"), _step("A2"), _step("A3")])
    parser = StubParser(
        plan=plan,
        decisions=[ReplanDecision(action="replan", reason="收窄", steps=[_step("B1")])],
    )
    router = RecordingRouter()

    events = _run_agent(parser, router)

    assert len(router.executed) == 2, "旧计划第 1 步 + 新计划 1 步"
    assert events[-1]["type"] == "done"
    assert events[-2]["type"] == "report"


def test_replan_empty_new_plan_ends():
    """新计划为空：直接结束并产出 report + done。"""
    plan = DiagnosticPlan(steps=[_step("A1"), _step("A2")])
    parser = StubParser(
        plan=plan,
        decisions=[ReplanDecision(action="replan", reason="没有新步骤", steps=[])],
    )
    router = RecordingRouter()

    events = _run_agent(parser, router)

    assert [s.description for s in router.executed] == ["A1"], "空新计划不应再执行任何步骤"
    assert _events_of_type(events, "report")
    assert events[-1]["type"] == "done"


def test_replan_duplicate_completed_steps_filtered():
    """已完成步骤不会重复执行：新计划里的重复项被过滤，只执行新项。"""
    plan = DiagnosticPlan(steps=[_step("A1", args={"user_id": "1001"})])
    dup = _step("A1", args={"user_id": "1001"})
    fresh = _step("新方向B", tool="retrieve_knowledge")
    parser = StubParser(
        plan=plan,
        decisions=[ReplanDecision(action="replan", reason="补查", steps=[dup, fresh])],
    )
    router = RecordingRouter()

    _run_agent(parser, router)

    assert [s.description for s in router.executed] == ["A1", "新方向B"], "重复步骤应被过滤"


def test_max_iterations_force_end():
    """达到最大迭代次数：即使 LLM 一直 replan 也强制结束，步骤数不超过上限。"""
    plan = DiagnosticPlan(steps=[_step("A1")])
    loop_step = _step("永远新步骤", tool="retrieve_knowledge")
    parser = StubParser(
        plan=plan,
        decisions=[ReplanDecision(action="replan", reason="再来", steps=[loop_step])] * 20,
    )
    router = RecordingRouter()

    events = _run_agent(parser, router)

    assert len(router.executed) <= MAX_ITERATIONS, f"执行步数不得超过 {MAX_ITERATIONS}"
    assert _events_of_type(events, "report"), "达到上限后仍必须产出 report"
    assert events[-1]["type"] == "done"


def test_final_events_always_report_then_done():
    """无论路径如何，事件流最后两件事一定是 report、done。"""
    parser = StubParser(plan=DiagnosticPlan(steps=[_step("A1")]), decisions=[])
    router = RecordingRouter()
    events = _run_agent(parser, router)
    assert events[-2]["type"] == "report"
    assert events[-1]["type"] == "done"


# ----------------------------------------------------------------- 工具失败语义
def test_tool_failure_produces_structured_step_result():
    """工具调用失败：StepResult 携带 error_code / safe_error_message，不含原始异常文本。"""
    plan = DiagnosticPlan(steps=[_step("坏工具步骤"), _step("正常步骤")])
    parser = StubParser(plan=plan, decisions=[])
    router = RecordingRouter(fail_tools={"query_device_status"})

    events = _run_agent(parser, router)

    failed_events = [e for e in _events_of_type(events, "step") if e.get("data", {}).get("error_code")]
    assert failed_events, "失败步骤的事件应携带 error_code"
    ev = failed_events[0]
    assert ev["data"]["error_code"] == "TOOL_EXECUTION_FAILED"
    assert "工具调用失败" in ev["content"]
    assert "RuntimeError" not in ev["content"] and "Traceback" not in ev["content"], (
        "不得把异常字符串伪装成正常诊断结果"
    )


def test_tool_router_unavailable_tool():
    """非法工具名：路由层返回 TOOL_UNAVAILABLE（schema 已挡一层，这里验证纵深防御）。"""
    rogue = DiagnosticStep.model_construct(description="越权步骤", tool="rm_rf_everything", arguments={})
    router = ToolRouter()
    result = router.execute(rogue, user_query="扫地机不工作")
    assert result.success is False
    assert result.error_code == "TOOL_UNAVAILABLE"
    assert result.safe_error_message


def test_tool_router_missing_required_argument():
    """必填参数缺失且无法补全：返回 TOOL_ARGUMENT_INVALID。"""
    router = ToolRouter(specs={
        "query_error_code": ToolSpec(func=lambda error_code: error_code, required=("error_code",)),
    })
    step = _step("查错误码", tool="query_error_code")
    result = router.execute(step, user_query="扫地机不工作，没有错误码")
    assert result.success is False
    assert result.error_code == "TOOL_ARGUMENT_INVALID"


def test_tool_router_auto_fill_from_user_query():
    """query_error_code 参数缺失时从用户描述提取错误码。"""
    from agent.diagnostic.tool_router import _fill_error_code

    router = ToolRouter(specs={
        "query_error_code": ToolSpec(
            func=lambda error_code: f"错误码{error_code}的资料",
            required=("error_code",),
            auto_fill={"error_code": _fill_error_code},
        ),
    })
    step = _step("查错误码", tool="query_error_code")
    result = router.execute(step, user_query="扫地机报 E03 错误")
    assert result.success is True
    assert "E03" in result.content


def test_tool_router_catches_execution_exception():
    """底层函数抛异常：路由层捕获并返回安全错误信息。"""
    def boom(**kw):
        raise RuntimeError("数据库炸了：敏感信息")

    router = ToolRouter(specs={
        "query_device_status": ToolSpec(func=boom, required=("user_id",)),
    })
    step = _step("查状态", tool="query_device_status", args={"user_id": "1001"})
    result = router.execute(step, user_query="扫地机不工作")
    assert result.success is False
    assert result.error_code == "TOOL_EXECUTION_FAILED"
    assert "敏感信息" not in result.safe_error_message, "安全信息不得包含原始异常内容"


# ----------------------------------------------------------------- schema 校验
def test_diagnostic_step_rejects_illegal_tool():
    with pytest.raises(ValidationError):
        DiagnosticStep(description="恶意步骤", tool="os_system", arguments={})


def test_diagnostic_step_accepts_all_whitelisted_tools():
    for tool in ALLOWED_TOOLS:
        assert DiagnosticStep(description="步骤", tool=tool).tool == tool


def test_replan_decision_accepts_plan_alias_and_normalizes_action():
    decision = ReplanDecision.model_validate({
        "action": "END",
        "reason": "够了",
        "plan": [{"description": "步骤", "tool": "retrieve_knowledge", "arguments": {}}],
    })
    assert decision.action == "end"
    assert decision.steps and decision.steps[0].tool == "retrieve_knowledge"


# ----------------------------------------------------------------- 解析层
class _Msg:
    def __init__(self, content):
        self.content = content


class TextModel:
    """invoke 返回固定文本；不支持 structured output。"""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def invoke(self, messages, **kw):
        self.calls += 1
        return _Msg(self.text)


def test_parser_structured_output_preferred():
    """模型支持 with_structured_output 时直接返回 schema 实例。"""

    class StructuredModel:
        def with_structured_output(self, schema):
            class Runner:
                def invoke(self, messages, **kw):
                    return schema(steps=[_step("结构化步骤")])

            return Runner()

    parser = LlmParser(model=StructuredModel())
    plan = parser.parse_plan("sys", "user")
    assert isinstance(plan, DiagnosticPlan)
    assert plan.steps[0].description == "结构化步骤"


def test_parser_fallback_extracts_json_with_aliases():
    model = TextModel('{"plan": ["字符串步骤"], "extra": 1}')
    parser = LlmParser(model=model)
    plan = parser.parse_plan("sys", "user")
    assert plan is not None
    assert plan.steps[0].description == "字符串步骤"
    assert plan.steps[0].tool == "retrieve_knowledge", "字符串步骤降级为全库检索"


def test_parser_fallback_accepts_bare_array():
    model = TextModel('["步骤一", "步骤二"]')
    parser = LlmParser(model=model)
    plan = parser.parse_plan("sys", "user")
    assert plan is not None and len(plan.steps) == 2


def test_parser_illegal_tool_in_plan_rejected():
    """计划里含非法工具名：只丢弃非法步骤，合法步骤保留（不再整体报废走兜底）。"""
    model = TextModel(json.dumps({
        "steps": [
            {"description": "正常", "tool": "retrieve_knowledge", "arguments": {}},
            {"description": "越权", "tool": "hack_tool", "arguments": {}},
            {"description": "也正常", "tool": "query_maintenance", "arguments": {}},
        ]
    }))
    parser = LlmParser(model=model)
    plan = parser.parse_plan("sys", "user")
    assert plan is not None, "存在合法步骤时不得整体丢弃"
    assert [s.description for s in plan.steps] == ["正常", "也正常"]
    assert all(s.tool in ALLOWED_TOOLS for s in plan.steps)


def test_parser_all_illegal_tools_returns_empty_plan():
    """计划里全部是非法工具名：无合法步骤可保留 → 返回空计划（planner 据此走兜底计划）。"""
    model = TextModel(json.dumps({
        "steps": [
            {"description": "越权1", "tool": "hack_tool", "arguments": {}},
            {"description": "越权2", "tool": "rm_rf", "arguments": {}},
        ]
    }))
    parser = LlmParser(model=model)
    plan = parser.parse_plan("sys", "user")
    assert plan is not None and plan.steps == [], "全部非法时应得到空计划"
    # planner 对空计划的既有行为：走固定兜底计划
    from agent.diagnostic.nodes import planner_node

    state = {"user_query": "扫地机不工作", "events": [], "iteration_count": 0}
    out = planner_node(state, parser=StubParser(plan=plan))
    assert out["pending_steps"], "空计划应触发 planner 兜底计划，流程不中断"


def test_parser_replan_keeps_valid_steps_drops_illegal():
    """重规划决策里的非法工具步骤同样只丢弃自身，合法步骤保留。"""
    model = TextModel(json.dumps({
        "action": "replan",
        "reason": "补充检索",
        "steps": [
            {"description": "合法检索", "tool": "retrieve_knowledge", "arguments": {}},
            {"description": "非法", "tool": "hack_tool", "arguments": {}},
        ],
    }))
    parser = LlmParser(model=model)
    decision = parser.parse_replan("sys", "user")
    assert decision is not None
    assert decision.action == "replan"
    assert [s.description for s in (decision.steps or [])] == ["合法检索"]


def test_parser_garbage_text_returns_none():
    parser = LlmParser(model=TextModel("这不是JSON，只是闲聊。"))
    assert parser.parse_plan("sys", "user") is None
    assert parser.parse_replan("sys", "user") is None


def test_parser_planner_fallback_on_none(monkeypatch):
    """planner 拿到 None 时使用固定兜底计划，流程不中断。"""
    from agent.diagnostic.nodes import planner_node

    state = {"user_query": "扫地机不工作", "events": [], "iteration_count": 0}
    out = planner_node(state, parser=StubParser(plan=None))
    steps = out["pending_steps"]
    assert steps and all(s.tool in ALLOWED_TOOLS for s in steps)
    assert out["events"][-1]["type"] == "plan"


def test_planner_truncates_overlong_plan():
    from agent.diagnostic.nodes import planner_node

    long_plan = DiagnosticPlan(steps=[_step(f"步骤{i}") for i in range(10)])
    state = {"user_query": "扫地机不工作", "events": [], "iteration_count": 0}
    out = planner_node(state, parser=StubParser(plan=long_plan))
    assert len(out["pending_steps"]) == 5, "计划步数应截断到 MAX_STEPS"


# ----------------------------------------------------------------- 兼容导入
def test_legacy_import_path_still_works():
    """agent.diagnostic_agent 兼容入口：旧调用方（orchestrator/scripts）无需修改。"""
    from agent.diagnostic_agent import (  # noqa: F401
        AGENT_NAME,
        DiagnosticAgent,
        executor_node,
        get_diagnostic_agent,
        planner_node,
        replanner_node,
        reporter_node,
    )


def test_executor_consumes_only_first_pending_step():
    from agent.diagnostic.nodes import executor_node

    state = {
        "user_query": "扫地机不工作",
        "pending_steps": [_step("A1"), _step("A2")],
        "completed_steps": [],
        "events": [],
    }
    out = executor_node(state, tool_router=RecordingRouter())
    assert [s.description for s in out["pending_steps"]] == ["A2"]
    assert out["completed_steps"][0].step.description == "A1"


def test_executor_empty_pending_ends():
    from agent.diagnostic.nodes import executor_node

    state = {"user_query": "扫地机不工作", "pending_steps": [], "completed_steps": [], "events": []}
    out = executor_node(state, tool_router=RecordingRouter())
    assert out["should_end"] is True


def test_reporter_fallback_report_without_llm():
    """reporter 的 LLM 不可用时输出结构化兜底报告（区分事实/工具不可用/推测）。"""

    class BoomModel:
        def invoke(self, messages, **kw):
            raise RuntimeError("模型挂了")

    from agent.diagnostic.nodes import reporter_node

    completed = [
        CompletedStep(step=_step("成功步骤"), result=StepResult(success=True, content="覆盖率 92%")),
        CompletedStep(
            step=_step("失败步骤"),
            result=StepResult(success=False, error_code="TOOL_EXECUTION_FAILED", safe_error_message="工具调用失败，本步骤已跳过。"),
        ),
    ]
    state = {"user_query": "扫地机不工作", "completed_steps": completed, "events": []}
    out = reporter_node(state, model=BoomModel())
    report = out["final_report"]
    assert "诊断报告" in report
    assert "覆盖率 92%" in report
    assert "工具不可用" in report or "调用失败" in report
    assert "推测性建议" in report
    assert "模型挂了" not in report, "兜底报告不得包含原始异常文本"


# ----------------------------------------------------------------- 失败路径加固（P0 任务 3）

def _all_events_text(events) -> str:
    return json.dumps(events, ensure_ascii=False, default=str)


def test_reporter_model_init_failure_still_report_and_done(monkeypatch):
    """get_chat_model() 初始化抛异常：仍产生且仅一个 report + done，异常文本不泄漏。"""

    def boom():
        raise RuntimeError("模型初始化失败：API_KEY=sk-secret")

    monkeypatch.setattr("agent.diagnostic.nodes.get_chat_model", boom)
    from agent.diagnostic.service import DiagnosticAgent

    parser = StubParser(plan=DiagnosticPlan(steps=[_step("A1")]), decisions=[])
    agent = DiagnosticAgent(parser=parser, tool_router=RecordingRouter(), model=None)
    events = list(agent.run("扫地机不工作"))

    reports = _events_of_type(events, "report")
    assert len(reports) == 1, "模型初始化失败也必须产生一个 report"
    assert "诊断报告" in reports[0]["content"], "应为结构化兜底报告"
    assert events[-1]["type"] == "done"
    assert "sk-secret" not in _all_events_text(events), "原始异常文本不得出现在任何事件中"


def test_reporter_invoke_failure_stream_level():
    """reporter.invoke 抛异常（流级）：仍产生 report + done，异常文本不泄漏。"""

    class InvokeBoomModel:
        def invoke(self, messages, **kw):
            raise RuntimeError("LLM 网关 503：内部凭证 leak-token")

    from agent.diagnostic.service import DiagnosticAgent

    parser = StubParser(plan=DiagnosticPlan(steps=[_step("A1")]), decisions=[])
    agent = DiagnosticAgent(parser=parser, tool_router=RecordingRouter(), model=InvokeBoomModel())
    events = list(agent.run("扫地机不工作"))

    assert len(_events_of_type(events, "report")) == 1
    assert events[-2]["type"] == "report"
    assert events[-1]["type"] == "done"
    assert sum(1 for e in events if e["type"] == "done") == 1
    assert "leak-token" not in _all_events_text(events)


def test_graph_stream_error_single_error_then_done(monkeypatch):
    """graph.stream 抛异常（无法降级）：只产生一个 error + 一个 done，done 最后。"""

    class BoomGraph:
        def stream(self, initial, stream_mode=None):
            raise RuntimeError("LangGraph 内部状态损坏：detail-secret")

    monkeypatch.setattr(
        "agent.diagnostic.service.build_diagnostic_graph",
        lambda **kwargs: BoomGraph(),
    )
    from agent.diagnostic.service import DiagnosticAgent

    agent = DiagnosticAgent()
    events = list(agent.run("扫地机不工作"))

    assert [e["type"] for e in events] == ["error", "done"]
    assert events[0]["data"]["error_code"] == "INTERNAL_ERROR"
    assert events[-1]["type"] == "done"
    assert "detail-secret" not in _all_events_text(events)


def test_generator_early_close_terminates_cleanly():
    """客户端提前关闭生成器：干净终止，不再发出任何事件（done 不会重复或补发）。"""
    from agent.diagnostic.service import DiagnosticAgent

    parser = StubParser(plan=DiagnosticPlan(steps=[_step("A1")]), decisions=[])
    agent = DiagnosticAgent(parser=parser, tool_router=RecordingRouter(), model=StubReportModel())
    gen = agent.run("扫地机不工作")
    first = next(gen)
    assert first["type"] == "plan"
    gen.close()  # 不应抛异常
    assert gen.gi_frame is None, "生成器应已正常关闭"


def test_tool_router_service_unavailable_error():
    """底层服务抛 ServiceUnavailableError：转为 success=False + 稳定错误码。"""
    from utils.exceptions import ServiceUnavailableError

    def unavailable(**kw):
        raise ServiceUnavailableError("实时设备数据不可用")

    router = ToolRouter(specs={
        "query_device_status": ToolSpec(func=unavailable, required=("user_id",)),
    })
    step = _step("查状态", tool="query_device_status", args={"user_id": "1001"})
    result = router.execute(step, user_query="扫地机不工作")
    assert result.success is False
    assert result.error_code == "SERVICE_UNAVAILABLE"
    assert "底层服务暂时不可用" in result.safe_error_message
    assert "实时设备数据不可用" not in result.safe_error_message, "原始异常文本不得进入安全提示"


def test_device_status_data_missing_not_recorded_as_confirmed_fact():
    """设备数据不可用（如用户不存在）：不得包装成 success=True 进入已确认事实。"""
    from agent.diagnostic.service import DiagnosticAgent

    plan = DiagnosticPlan(steps=[_step("查询设备运行状态", args={"user_id": "999999"})])
    parser = StubParser(plan=plan, decisions=[])

    class InvokeBoomModel:
        def invoke(self, messages, **kw):
            raise RuntimeError("llm-down")

    agent = DiagnosticAgent(parser=parser, tool_router=ToolRouter(), model=InvokeBoomModel())
    events = list(agent.run("扫地机不工作"))

    step_events = _events_of_type(events, "step")
    assert step_events, "应执行了设备状态步骤"
    ev = step_events[0]
    assert ev["data"]["error_code"] == "SERVICE_UNAVAILABLE", (
        "用户数据缺失应映射为 SERVICE_UNAVAILABLE，而不是成功结果"
    )
    assert "底层服务暂时不可用" in ev["content"], "step 事件内容应为安全提示"
    report = _events_of_type(events, "report")[0]["content"]
    assert "工具不可用" in report, "兜底报告应把失败步骤归入工具不可用"
    assert "未查询到用户999999" not in report, "底层错误字符串不得进入报告"


def test_planner_dedupes_duplicate_steps():
    """计划内部重复步骤：稳定去重（首见保留），不影响其他步骤。"""
    from agent.diagnostic.nodes import planner_node

    plan = DiagnosticPlan(steps=[
        _step("查状态", args={"user_id": "1001"}),
        _step("查状态", args={"user_id": "1001"}),
        _step("查资料", tool="retrieve_knowledge"),
    ])
    state = {"user_query": "扫地机不工作", "events": [], "iteration_count": 0}
    out = planner_node(state, parser=StubParser(plan=plan))
    assert [s.description for s in out["pending_steps"]] == ["查状态", "查资料"]


def test_replanner_truncates_overlong_new_plan():
    """replan 新计划超过 MAX_STEPS：截断到上限。"""
    from agent.diagnostic.nodes import replanner_node

    completed = [CompletedStep(step=_step("A1"), result=StepResult(success=True, content="ok"))]
    new_steps = [_step(f"新{i}", tool="retrieve_knowledge") for i in range(10)]
    state = {
        "user_query": "扫地机不工作",
        "pending_steps": [],
        "completed_steps": completed,
        "iteration_count": 1,
        "events": [],
    }
    decision = ReplanDecision(action="replan", reason="扩大排查", steps=new_steps)
    out = replanner_node(state, parser=StubParser(plan=None, decisions=[decision]))
    assert len(out["pending_steps"]) == MAX_STEPS, f"新计划应截断到 {MAX_STEPS}"


# ----------------------------------------------------------------- 工具输入输出安全边界（P1-7）
def test_failed_step_result_not_in_reporter_confirmed_facts():
    """失败步骤保持 success=False：其内容不进入 Reporter 的「已确认事实」段落。"""
    from agent.diagnostic.nodes import reporter_node

    failed = StepResult(
        success=False,
        error_code="TOOL_EXECUTION_FAILED",
        safe_error_message="步骤「查状态」的工具调用失败，本步骤已跳过。",
    )
    ok = StepResult(success=True, content="设备覆盖率 92%，一切正常")
    state = {
        "user_query": "扫地机不工作",
        "pending_steps": [],
        "completed_steps": [
            CompletedStep(step=_step("查状态"), result=failed),
            CompletedStep(step=_step("查资料", tool="retrieve_knowledge"), result=ok),
        ],
        "iteration_count": 1,
        "events": [],
    }

    class ReportModel:
        def invoke(self, messages, **kw):
            captured.append(messages)
            class R:
                content = "## 诊断报告"
            return R()

    captured: list = []
    out = reporter_node(state, model=ReportModel())
    report_event = next(e for e in out["events"] if e["type"] == "report")

    context_text = str(captured[0])
    assert "设备覆盖率 92%" in context_text, "成功步骤内容应进入报告上下文"
    # 失败步骤的降级文案只以「工具不可用/调用失败」标注出现，绝不伪装成事实
    assert "（工具不可用/调用失败" in context_text, "失败步骤必须带非事实标注"
    failure_line = [ln for ln in context_text.split("\n") if "工具不可用/调用失败" in ln]
    assert len(failure_line) == 1, "失败步骤只出现一次"
    assert "诊断报告" in report_event["content"]


def test_fallback_report_confirmed_facts_exclude_failed_steps():
    """兜底报告：只有 success=True 的步骤进入「已确认事实」，失败步骤单列为不可用。"""
    from agent.diagnostic.nodes import _fallback_report

    failed = StepResult(
        success=False,
        error_code="SERVICE_UNAVAILABLE",
        safe_error_message="步骤「查状态」的底层服务暂时不可用，本步骤已跳过。",
    )
    ok = StepResult(success=True, content="电池电压偏低，建议更换")
    report = _fallback_report("扫地机不工作", [
        CompletedStep(step=_step("查状态"), result=failed),
        CompletedStep(step=_step("查资料", tool="retrieve_knowledge"), result=ok),
    ])

    assert "电池电压偏低" in report, "成功步骤内容进入排查过程"
    assert "工具不可用/调用失败" in report, "失败步骤带明确标注"
    assert "以下步骤的工具不可用" in report, "失败步骤单列为不可用清单"
    assert "基于以上已确认的排查数据" in report, "存在已确认事实时的原因表述"


def test_tool_router_logs_cleaned_arguments_on_all_paths(caplog):
    """tool_router 三类日志（成功 / 执行异常 / 服务不可用）均使用清洗后的参数。"""
    import logging

    from agent.diagnostic.tool_router import ToolRouter, ToolSpec

    def ok(**kw):
        return "正常结果"

    def boom(**kw):
        raise RuntimeError("数据库炸了：api_key=sk-ROUTER-998877665544")

    def unavailable(**kw):
        from utils.exceptions import ServiceUnavailableError
        raise ServiceUnavailableError("设备数据服务不可用")

    router = ToolRouter(specs={
        "query_device_status": ToolSpec(func=ok, required=("user_id",)),
        "query_maintenance": ToolSpec(func=boom, required=("query",)),
        "retrieve_knowledge": ToolSpec(func=unavailable, required=("query",)),
    })

    secret_arg = "api_key=sk-ARG-998877665544"
    with caplog.at_level(logging.INFO, logger="agent"):
        r1 = router.execute(_step("查状态", args={"user_id": secret_arg}), user_query="q")
        r2 = router.execute(_step("查维护", tool="query_maintenance", args={"query": secret_arg}), user_query="q")
        r3 = router.execute(_step("查资料", tool="retrieve_knowledge", args={"query": secret_arg}), user_query="q")

    assert r1.success is True and r1.content == "正常结果", "正常调用行为不变"
    assert r2.success is False and r2.error_code == "TOOL_EXECUTION_FAILED"
    assert r3.success is False and r3.error_code == "SERVICE_UNAVAILABLE"

    for event_name in ("tool_router_success", "tool_router_execution_error", "tool_router_service_unavailable"):
        entries = [
            r.msg for r in caplog.records
            if isinstance(r.msg, dict) and r.msg.get("event") == event_name
        ]
        assert entries, f"应记录 {event_name} 日志"
        serialized = str(entries[-1])
        assert "sk-ARG-998877665544" not in serialized, f"{event_name} 日志不得含原始密钥"
        assert "sk-ROUTER-998877665544" not in serialized, f"{event_name} 日志不得含异常原文密钥"
        assert "api_key=***REDACTED***" in serialized, f"{event_name} 日志应含脱敏标记"


# ----------------------------------------------------------------- Reporter 异常日志收敛（P1-7.1）
def test_reporter_error_log_safe_and_fallback_intact(caplog):
    """Reporter 异常含裸密钥 / 绝对路径 / 用户输入：日志只留清洗摘要，fallback 与事实链不变。"""
    import logging

    from agent.diagnostic.service import DiagnosticAgent

    secret = "sk-REPORTER-998877665544"
    abs_path = "D:\\secret\\models\\reporter\\prompt.bin"
    user_input = "用户私密故障描述"
    # 填充段保证路径与用户输入落在 100 字截断边界之后（密钥在前，先脱敏再截断）
    filler = "报告模型连接失败，已重试多次仍无法恢复服务。" * 4

    class LeakyModel:
        def invoke(self, messages, **kw):
            raise RuntimeError(f"api_key={secret} {filler} 路径 {abs_path} 输入[{user_input}]")

    parser = StubParser(
        plan=DiagnosticPlan(steps=[_step("查状态"), _step("查资料", tool="retrieve_knowledge")]),
        # 第一步后 continue 执行第二步，随后默认 end 进入报告
        decisions=[ReplanDecision(action="continue", reason="继续下一步")],
    )
    router = RecordingRouter(results=[
        StepResult(success=True, content="覆盖率 92%"),
        StepResult(
            success=False,
            error_code="TOOL_EXECUTION_FAILED",
            safe_error_message="步骤「查资料」的工具调用失败，本步骤已跳过。",
        ),
    ])

    # 直接构造（model 注入 LeakyModel）：_run_agent 固定用 StubReportModel
    agent = DiagnosticAgent(parser=parser, tool_router=router, model=LeakyModel())
    with caplog.at_level(logging.ERROR, logger="agent"):
        events = list(agent.run("扫地机不工作"))

    # fallback 行为不变：仍生成结构化兜底报告，report + done 各一次
    reports = _events_of_type(events, "report")
    assert len(reports) == 1, "仍产生且仅一个 report 事件"
    assert events[-1]["type"] == "done", "仍以 done 结束"
    report = reports[0]["content"]
    assert "诊断报告" in report
    assert "覆盖率 92%" in report, "成功步骤事实保留（事实链不受影响）"
    assert "工具不可用/调用失败" in report, "失败步骤标注保留"
    all_text = _all_events_text(events)
    for leak in (secret, abs_path, user_input):
        assert leak not in all_text, f"事件流不得泄漏：{leak}"

    # 日志：统一摘要字段（error_type + 清洗后的 error_msg），无泄漏
    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "reporter_error"
    ]
    assert entries, "应记录 reporter_error 日志"
    entry = entries[-1]
    assert entry["error_type"] == "RuntimeError", "保留 error_type"
    assert "api_key=***REDACTED***" in entry["error_msg"], "前部密钥脱敏后保留标记"
    assert "…(共" in entry["error_msg"], "超长异常文本应被截断"

    serialized = str(entry)
    assert secret not in serialized, "原始密钥不得进日志"
    assert abs_path not in serialized, "完整绝对路径不得进日志"
    assert user_input not in serialized, "用户输入不得进日志"
    assert "traceback" not in entry, "不得记录 traceback 字段"

    record = next(r for r in caplog.records
                  if isinstance(r.msg, dict) and r.msg.get("event") == "reporter_error")
    assert record.exc_info is None, "禁止 exc_info=True"


# ----------------------------------------------------------------- 普通日志输入清洗（P1-8）
_LEAK_SECRET = "sk-NODES-998877665544"
_LEAK_PATH = "D:\\secret\\internal\\planner_cache.bin"
_LONG_DESC = "检查电源连接情况" + "x" * 300


def _leak_records(caplog, event_name):
    return [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == event_name
    ]


def test_planner_and_executor_logs_sanitized_protocol_intact(caplog):
    """planner_done / executor_step / replanner_new_plan / replanner_end：日志清洗，协议不变。"""
    import logging

    leak_desc = f"读取配置 {_LEAK_PATH}，凭据 api_key={_LEAK_SECRET}"
    leak_reason = f"需要补充检查，密钥 {_LEAK_SECRET}，路径 {_LEAK_PATH}"
    new_step_desc = f"新计划步骤：核对 {_LEAK_PATH}"

    plan = DiagnosticPlan(steps=[_step(leak_desc), _step(_LONG_DESC, tool="retrieve_knowledge")])
    parser = StubParser(
        plan=plan,
        decisions=[ReplanDecision(action="replan", reason=leak_reason, steps=[_step(new_step_desc)])],
    )
    router = RecordingRouter()

    with caplog.at_level(logging.INFO, logger="agent"):
        events = _run_agent(parser, router)

    # 行为不变：report + done 各一次，执行顺序符合 replan 语义
    assert len(_events_of_type(events, "report")) == 1
    assert events[-1]["type"] == "done"
    assert [s.description for s in router.executed] == [leak_desc, new_step_desc]

    # 事件流协议不变：plan / step 事件保留原始文本（前端协议，不受日志清洗影响）
    plan_ev = _events_of_type(events, "plan")[0]
    assert leak_desc in plan_ev["data"]["steps"]
    step_evs = _events_of_type(events, "step")
    assert leak_desc in step_evs[0]["data"]["description"]

    # planner_done：字段结构与事件名不变，内容清洗
    planner_logs = _leak_records(caplog, "planner_done")
    assert planner_logs, "应记录 planner_done"
    entry = planner_logs[-1]
    assert set(entry) == {"event", "steps"}, "planner_done 字段结构不变"
    assert isinstance(entry["steps"], list)
    serialized = str(entry)
    assert _LEAK_SECRET not in serialized, "planner 日志不得含原始密钥"
    assert _LEAK_PATH not in serialized, "planner 日志不得含绝对路径"
    assert _LONG_DESC not in serialized, "超长描述不得完整进日志"
    assert any(s.endswith("…(共308字)") for s in entry["steps"] if isinstance(s, str)), "超长描述截断"

    # executor_step：字段结构不变（event/step/tool/success/error_code），step 清洗
    exec_logs = _leak_records(caplog, "executor_step")
    assert len(exec_logs) == 2, "两次执行各记一条"
    for log_entry in exec_logs:
        assert set(log_entry) == {"event", "step", "tool", "success", "error_code"}, "executor_step 字段结构不变"
        assert log_entry["tool"] == "query_device_status", "工具名保留"
        assert isinstance(log_entry["success"], bool)
        assert _LEAK_SECRET not in str(log_entry), "executor 日志不得含原始密钥"
        assert _LEAK_PATH not in str(log_entry), "executor 日志不得含绝对路径"

    # replanner_new_plan：new_plan + reason 清洗
    new_plan_logs = _leak_records(caplog, "replanner_new_plan")
    assert new_plan_logs
    entry = new_plan_logs[-1]
    assert set(entry) == {"event", "new_plan", "reason"}, "replanner_new_plan 字段结构不变"
    serialized = str(entry)
    assert _LEAK_SECRET not in serialized
    assert _LEAK_PATH not in serialized
    assert "***REDACTED***" in serialized, "密钥以脱敏标记呈现"

    # replanner_end：reason 清洗（默认 end 决策 reason 无泄漏，仅验证结构）
    end_logs = _leak_records(caplog, "replanner_end")
    assert end_logs, "应记录 replanner_end"
    assert set(end_logs[-1]) == {"event", "reason"}

    # 全部日志无 traceback / exc_info
    for r in caplog.records:
        assert r.exc_info is None, "普通业务日志禁止 exc_info"


def test_replanner_continue_and_end_reason_sanitized(caplog):
    """replanner_continue / replanner_end 的 reason 含泄漏时清洗，行为不变。"""
    import logging

    leak_reason = f"继续排查，密钥 {_LEAK_SECRET}，读取 {_LEAK_PATH}"
    end_reason = f"信息足够，密钥 {_LEAK_SECRET}"

    plan = DiagnosticPlan(steps=[_step("步骤一"), _step("步骤二")])
    parser = StubParser(
        plan=plan,
        decisions=[
            ReplanDecision(action="continue", reason=leak_reason),
            ReplanDecision(action="end", reason=end_reason),
        ],
    )
    router = RecordingRouter()

    with caplog.at_level(logging.INFO, logger="agent"):
        events = _run_agent(parser, router)

    # 行为不变：两步都执行，report + done 正常收尾
    assert len(router.executed) == 2
    assert len(_events_of_type(events, "report")) == 1
    assert events[-1]["type"] == "done"

    cont_logs = _leak_records(caplog, "replanner_continue")
    assert cont_logs, "应记录 replanner_continue"
    entry = cont_logs[-1]
    assert set(entry) == {"event", "reason"}, "replanner_continue 字段结构不变"
    serialized = str(entry)
    assert _LEAK_SECRET not in serialized, "原始密钥不得进日志"
    assert _LEAK_PATH not in serialized, "绝对路径不得进日志"
    assert "***REDACTED***" in serialized, "密钥以脱敏标记呈现"

    end_logs = _leak_records(caplog, "replanner_end")
    assert end_logs
    entry = end_logs[-1]
    assert set(entry) == {"event", "reason"}
    assert _LEAK_SECRET not in str(entry), "end 决策密钥不得进日志"
    assert "***REDACTED***" in str(entry)

    # 事件流协议不变：replan 事件保留原始 reason 文本
    replan_evs = _events_of_type(events, "replan")
    assert any(leak_reason in e["content"] for e in replan_evs), "事件内容不受日志清洗影响"

    for r in caplog.records:
        assert r.exc_info is None


# ----------------------------------------------------------------- 工具返回值安全边界（P1-11）
def test_tool_router_exception_full_spectrum_no_leak_into_events():
    """工具异常含密钥 / 路径 / 用户输入 / 远端响应：StepResult 与事件流零泄漏。"""
    from agent.diagnostic.service import DiagnosticAgent

    secret = "sk-ROUTER-998877665544"
    abs_path = "D:\\secret\\router\\device.bin"
    user_text = "用户输入原文扫地机充不进电"
    remote_body = '{"status": 500, "internal": "远端原始响应"}'

    def boom(**kw):
        raise RuntimeError(
            f"底层调用失败 api_key={secret} 路径 {abs_path} "
            f"echo={user_text} resp={remote_body}"
        )

    router = ToolRouter(specs={
        "query_device_status": ToolSpec(func=boom, required=("user_id",)),
    })
    step = _step("查询设备运行状态", tool="query_device_status", args={"user_id": "1001"})
    result = router.execute(step, user_query=user_text)

    # StepResult：失败语义 + 固定模板文案，无任何原始异常成分
    assert result.success is False
    assert result.error_code == "TOOL_EXECUTION_FAILED"
    assert result.content == "", "失败结果不得携带 content"
    assert "本步骤已跳过" in result.safe_error_message
    for leaked in (secret, abs_path, user_text, remote_body, "RuntimeError", "底层调用失败"):
        assert leaked not in result.safe_error_message, f"安全提示不得包含：{leaked[:30]}"

    # 完整事件流：step 事件内容（即 safe_error_message）与全部事件零泄漏
    class BoomModel:
        def invoke(self, messages, **kw):
            raise RuntimeError("llm-down")  # 触发 fallback 报告，不加载真实模型

    plan = DiagnosticPlan(steps=[_step("查询设备运行状态", args={"user_id": "1001"})])
    agent = DiagnosticAgent(
        parser=StubParser(plan=plan, decisions=[]),
        tool_router=router,
        model=BoomModel(),
    )
    events = list(agent.run(user_text))

    serialized = json.dumps(events, ensure_ascii=False, default=str)
    for leaked in (secret, abs_path, remote_body, "RuntimeError"):
        assert leaked not in serialized, f"事件流不得泄漏：{leaked[:30]}"

    step_events = _events_of_type(events, "step")
    assert step_events and step_events[0]["data"]["error_code"] == "TOOL_EXECUTION_FAILED"
    assert "本步骤已跳过" in step_events[0]["content"], "事件内容为模板化安全文案"

    report = _events_of_type(events, "report")[0]["content"]
    assert secret not in report and abs_path not in report, "报告不得泄漏异常细节"
    assert user_text not in report or "工具不可用" in report, "失败步骤不得进入已确认事实"
