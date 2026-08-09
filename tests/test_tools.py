"""诊断工具测试：设备状态（CSV）、RAG 类工具（桩知识库）、_safe_call 兜底。"""
import pytest

from agent.tools import diagnostic_tools as dt


class _KnowledgeStub:
    def search_troubleshooting(self, q):
        return "错误码E01：检查电源适配器是否插紧。"

    def search_maintenance(self, q):
        return "维护建议：每三个月清理一次滤网。"

    def retrieve(self, q, source_files=None):
        return "通用知识：参见故障排除手册。"


@pytest.fixture
def stub_knowledge(monkeypatch):
    monkeypatch.setattr("agent.knowledge_agent.get_knowledge_agent", lambda: _KnowledgeStub())


def test_query_device_status_from_csv():
    result = dt.query_device_status.func("1006")
    assert isinstance(result, str)
    assert len(result) > 0, "设备状态查询应返回非空字符串"


def test_current_user_id_returns_string():
    uid = dt.current_user_id()
    assert isinstance(uid, str) and uid


def test_query_error_code_uses_knowledge(stub_knowledge):
    result = dt.query_error_code.func("E01")
    assert "E01" in result


def test_query_maintenance_uses_knowledge(stub_knowledge):
    result = dt.query_maintenance.func("如何保养")
    assert "滤网" in result


def test_retrieve_knowledge_uses_knowledge(stub_knowledge):
    result = dt.retrieve_knowledge.func("扫地机异响")
    assert "故障排除手册" in result


def test_safe_call_wraps_exception():
    def boom():
        raise RuntimeError("boom")

    out = dt._safe_call("t", boom)
    assert "失败" in out, "_safe_call 应捕获异常并返回失败提示"


def test_safe_call_passes_through():
    assert dt._safe_call("t", lambda: "ok") == "ok"
