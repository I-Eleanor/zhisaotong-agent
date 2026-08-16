"""Agent 行为测试：对话 Agent 事件翻译与多轮记忆、诊断 Agent 全流程、Orchestrator 路由。

对话 Agent 的流式执行依赖支持 bind_tools 的真实模型，这里只测其纯逻辑
（_build_input / _to_events）；诊断 Agent 通过打桩的假模型走完 plan→execute→replan→report。
"""
from agent.conversation_agent import ConversationAgent
from agent.diagnostic_agent import (
    DiagnosticAgent,
    executor_node,
    planner_node,
    replanner_node,
    reporter_node,
)
from agent.memory.conversation_buffer import ConversationBuffer
from agent.orchestrator import Orchestrator


# ---- 局部消息替身（仅用于 _to_events 的 class name 判定，无需真实 LangChain 消息）
class HumanMessage:
    pass


class ToolMessage:
    def __init__(self, content, name=""):
        self.content = content
        self.name = name


class AIMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


# ----------------------------------------------------------------- 对话 Agent 纯逻辑
def test_build_input_includes_history():
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ]
    out = ConversationAgent._build_input("新提问", history)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert out["messages"][-1]["content"] == "新提问"


def test_to_events_human_no_echo():
    assert ConversationAgent._to_events(None, HumanMessage()) == []


def test_to_events_ai_message():
    evs = ConversationAgent._to_events(None, AIMessage(content="你好"))
    assert len(evs) == 1 and evs[0]["type"] == "message" and "你好" in evs[0]["content"]


def test_to_events_tool_call_and_result():
    evs = ConversationAgent._to_events(None, AIMessage(tool_calls=[{"name": "rag_summarize", "args": {}}]))
    assert any(e["type"] == "tool_start" for e in evs)
    evs2 = ConversationAgent._to_events(None, ToolMessage(content="检索结果", name="rag_summarize"))
    assert evs2[0]["type"] == "tool_end" and "检索结果" in evs2[0]["content"]


def test_to_events_skip_history_echo():
    # 历史助手消息内容不应被回显为新的 message 事件（避免前端重复）
    evs = ConversationAgent._to_events(None, AIMessage(content="重复内容"), skip={"重复内容"})
    assert evs == []


# ----------------------------------------------------------------- 诊断 Agent 节点
def _base_state(**over):
    state = {
        "user_query": "扫地机不工作",
        "plan": [],
        "current_step_index": 0,
        "execution_results": [],
        "iteration_count": 0,
        "final_report": "",
        "should_end": False,
        "events": [],
    }
    state.update(over)
    return state


def test_planner_node_returns_plan():
    out = planner_node(_base_state())
    assert isinstance(out["plan"], list) and out["plan"]
    assert any("设备" in s for s in out["plan"])


def test_executor_node_device_status():
    out = executor_node(_base_state(plan=["查询设备运行状态"]))
    assert out["events"][0]["type"] == "step"
    assert out["execution_results"]


def test_replanner_node_end():
    out = replanner_node(_base_state(plan=["查询设备运行状态"], execution_results=["ok"]))
    assert out["should_end"] is True


def test_reporter_node_returns_report():
    out = reporter_node(_base_state(execution_results=["步骤1结果"]))
    assert "诊断报告" in out["final_report"]


def test_diagnostic_run_full_flow():
    agent = DiagnosticAgent()
    types = [ev["type"] for ev in agent.run("我的扫地机器人最近清洁效率很低")]
    assert "plan" in types
    assert "step" in types
    assert "report" in types
    assert types[-1] == "done"


# ----------------------------------------------------------------- Orchestrator 路由
def test_orchestrator_route():
    # 绕过 __init__，避免构造真实 Agent（其需要支持 bind_tools 的模型）
    orch = Orchestrator.__new__(Orchestrator)
    assert orch.route("扫地机不工作了") == "diagnostic"
    # 无关键词 → LLM 分类（假模型返回 conversation）
    assert orch.route("怎么清理滤网") == "conversation"


# ----------------------------------------------------------------- 多轮记忆
def test_conversation_buffer_summary_after_overflow():
    buf = ConversationBuffer(max_rounds=2)
    buf.add_user_message("怎么清理滤网")
    buf.add_assistant_message("用软布擦拭滤网。")
    buf.add_user_message("那多久换一次")
    buf.add_assistant_message("每三个月更换。")
    buf.add_user_message("电池怎么保养")
    buf.add_assistant_message("电池避免亏电。")

    msgs = buf.get_messages()
    assert any(m["role"] == "system" for m in msgs), "溢出后应注入摘要 system 消息"
    assert buf.summary, "摘要应非空"
    recent = [m for m in msgs if m["role"] in ("user", "assistant")]
    assert len(recent) == 4, "最近的 2 轮（4 条）应完整保留"


def test_get_history_for_query_excludes_last_user():
    buf = ConversationBuffer(max_rounds=10, summary_enabled=False)
    buf.add_user_message("q1")
    buf.add_assistant_message("a1")
    buf.add_user_message("q2")
    hist = buf.get_history_for_query()
    assert hist[-1]["role"] == "assistant", "最后一条 user 提问应被排除"


# ----------------------------------------------------------------- Orchestrator 关键字路由 + LLM 降级
def test_orchestrator_keyword_route_diagnostic():
    orch = Orchestrator.__new__(Orchestrator)
    for kw in ["不工作", "故障", "报错", "异响", "卡住", "充不进电", "error", "stuck"]:
        assert orch.route(kw) == "diagnostic", f"关键词「{kw}」应路由到 diagnostic"


def test_orchestrator_llm_fallback_on_failure(monkeypatch):
    orch = Orchestrator.__new__(Orchestrator)

    def boom(*a, **kw):
        raise RuntimeError("LLM 不可用")

    monkeypatch.setattr("model.factory.get_chat_model", boom)
    result = orch.route("你好世界")
    assert result == "conversation", "LLM 不可用时应降级到 conversation"


def test_orchestrator_llm_classify_diagnostic(monkeypatch):
    orch = Orchestrator.__new__(Orchestrator)

    class FakeModel:
        def invoke(self, messages, **kw):
            class R:
                content = "diagnostic"
            return R()

    monkeypatch.setattr("model.factory.get_chat_model", lambda: FakeModel())
    monkeypatch.setattr("agent.orchestrator.load_orchestrator_prompt", lambda: "test")
    result = orch.route("机器发出异响")
    assert result == "diagnostic"


# ----------------------------------------------------------------- Diagnostic 最大迭代保护
def test_diagnostic_max_iterations():
    agent = DiagnosticAgent()
    events = list(agent.run("扫地机不工作"))
    replan_events = [e for e in events if e["type"] == "replan"]
    assert len(replan_events) <= 5, "迭代次数不应超过 MAX_ITERATIONS(5)"


def test_planner_json_parse_fallback(monkeypatch):
    from agent.diagnostic_agent import _llm_json

    def bad_model(*a, **kw):
        raise RuntimeError("LLM 故意失败")

    monkeypatch.setattr("agent.diagnostic_agent.get_chat_model", bad_model)
    result = _llm_json("sys", "user")
    assert result is None, "LLM 失败时应返回 None，由 planner 兜底"


def test_extract_json_with_code_fence():
    from agent.diagnostic_agent import _extract_json
    assert _extract_json('```json\n{"plan": ["步骤1"]}\n```') == {"plan": ["步骤1"]}


def test_extract_json_bare():
    from agent.diagnostic_agent import _extract_json
    assert _extract_json('["步骤1", "步骤2"]') == ["步骤1", "步骤2"]


def test_extract_json_empty():
    from agent.diagnostic_agent import _extract_json
    assert _extract_json("") is None
    assert _extract_json("无JSON内容") is None


# ----------------------------------------------------------------- ConversationBuffer 截断
def test_buffer_no_compress_within_limit():
    buf = ConversationBuffer(max_rounds=5, summary_enabled=False)
    buf.add_user_message("q1")
    buf.add_assistant_message("a1")
    buf.add_user_message("q2")
    buf.add_assistant_message("a2")
    assert not buf.summary, "未超出 max_rounds 不应压缩"


def test_buffer_clear():
    buf = ConversationBuffer(max_rounds=5)
    buf.add_user_message("q1")
    buf.add_assistant_message("a1")
    buf.clear()
    assert buf.recent_messages == []
    assert buf.summary == ""


def test_buffer_add_empty_content_ignored():
    buf = ConversationBuffer(max_rounds=5)
    buf.add_message("user", "")
    assert buf.recent_messages == [], "空内容不应写入"
