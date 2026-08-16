"""事件模块测试：make_event / event_to_text / events_to_text 全分支覆盖。"""
from agent.events import (
    VALID_EVENT_TYPES,
    event_to_text,
    events_to_text,
    make_event,
)


def test_make_event_basic():
    e = make_event("message", "agent1", "hello")
    assert e["type"] == "message"
    assert e["agent"] == "agent1"
    assert e["content"] == "hello"


def test_make_event_with_data():
    e = make_event("route", "orch", "", mode="diagnostic", mode_label="设备诊断")
    assert e["data"]["mode"] == "diagnostic"


def test_make_event_no_data():
    e = make_event("done", "agent1", "")
    assert "data" not in e


def test_event_to_text_message():
    assert event_to_text({"type": "message", "content": "你好"}) == "你好"


def test_event_to_text_report():
    assert event_to_text({"type": "report", "content": "诊断结果"}) == "诊断结果"


def test_event_to_text_route():
    text = event_to_text({"type": "route", "content": "", "data": {"mode_label": "设备诊断"}})
    assert "[路由]" in text
    assert "设备诊断" in text


def test_event_to_text_route_fallback_content():
    text = event_to_text({"type": "route", "content": "对话问答"})
    assert "对话问答" in text


def test_event_to_text_plan():
    text = event_to_text({"type": "plan", "data": {"steps": ["步骤1", "步骤2"]}})
    assert "[排查计划]" in text
    assert "步骤1" in text


def test_event_to_text_step():
    text = event_to_text({"type": "step", "content": "结果", "data": {"index": 1, "description": "检查电源"}})
    assert "[步骤1]" in text
    assert "检查电源" in text


def test_event_to_text_replan():
    text = event_to_text({"type": "replan", "content": "需要重新规划"})
    assert "[重规划]" in text


def test_event_to_text_tool_start():
    text = event_to_text({"type": "tool_start", "data": {"tool": "query_device_status"}})
    assert "[调用工具]" in text
    assert "query_device_status" in text


def test_event_to_text_error():
    text = event_to_text({"type": "error", "content": "出错了"})
    assert "[错误]" in text


def test_event_to_text_unknown_type():
    assert event_to_text({"type": "unknown", "content": "fallback"}) == "fallback"


def test_event_to_text_empty_content():
    assert event_to_text({"type": "message", "content": ""}) == ""


def test_events_to_text_filters_empty():
    events = [
        {"type": "message", "content": ""},
        {"type": "message", "content": "hello"},
    ]
    result = list(events_to_text(iter(events)))
    assert result == ["hello"]


def test_valid_event_types():
    assert "message" in VALID_EVENT_TYPES
    assert "done" in VALID_EVENT_TYPES
    assert "route" in VALID_EVENT_TYPES
    assert len(VALID_EVENT_TYPES) == 10
