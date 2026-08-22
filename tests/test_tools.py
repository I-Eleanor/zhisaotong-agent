"""诊断工具测试：设备状态（CSV）、RAG 类工具（桩知识库）、_safe_call 兜底。"""
import pytest

from agent.tools import diagnostic_tools as dt


class _KnowledgeStub:
    def search_troubleshooting_strict(self, q):
        return "错误码E01：检查电源适配器是否插紧。"

    def search_troubleshooting(self, q):
        return self.search_troubleshooting_strict(q)

    def search_maintenance_strict(self, q):
        return "维护建议：每三个月清理一次滤网。"

    def search_maintenance(self, q):
        return self.search_maintenance_strict(q)

    def retrieve_strict(self, q, source_files=None):
        return "通用知识：参见故障排除手册。"

    def retrieve(self, q, source_files=None):
        return self.retrieve_strict(q, source_files)


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


# ----------------------------------------------------------------- 工具安全边界（P1-7）
SECRET = "sk-TOOLBOUNDARY-1122334455"
USER_INPUT = "用户的私密提问内容"
ABS_PATH = "D:\\secret\\internal\\device_data.csv"


def _boom_with_sensitive(*args, **kwargs):
    raise RuntimeError(f"读取 {ABS_PATH} 失败：api_key={SECRET} 输入[{USER_INPUT}]")


def test_safe_call_returns_fixed_message_without_leak():
    """异常含裸密钥 / 绝对路径 / 用户输入时，工具返回值不泄漏（两套模块语义一致）。"""
    from agent.tools import agent_tools as at

    for module, name in ((dt, "diagnostic"), (at, "conversation")):
        out = module._safe_call("t", _boom_with_sensitive, "查询")
        assert out == "工具调用失败，请稍后重试", f"{name} 工具应返回固定安全文案，实际：{out!r}"
        for leak in (SECRET, ABS_PATH, USER_INPUT, "RuntimeError"):
            assert leak not in out, f"{name} 工具返回值泄漏：{leak}"


def test_safe_call_error_log_cleans_args_and_secret(caplog):
    """失败日志：args/kwargs 与异常文本都清洗，不含原始密钥与超长文本。"""
    import logging

    long_arg = "超长参数" + "y" * 300
    with caplog.at_level(logging.ERROR, logger="agent"):
        out = dt._safe_call("t", _boom_with_sensitive, long_arg, filter="api_key=sk-KWARG-998877665544")

    assert out == "工具调用失败，请稍后重试"
    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "diagnostic_tool_error"
    ]
    assert entries
    entry = entries[-1]
    serialized = str(entry)
    assert SECRET not in serialized, "异常原文中的密钥不得进日志"
    assert "sk-KWARG-998877665544" not in serialized, "kwargs 中的密钥不得进日志"
    assert ABS_PATH not in serialized, "绝对路径不得进日志"
    assert "y" * 200 not in serialized, "超长参数不得完整进日志"
    assert entry["args"][0].endswith("…(共304字)"), "超长参数被截断"


def test_safe_call_success_log_cleans_args(caplog):
    """成功日志同样清洗：参数中的密钥与超长文本不进日志，返回值与传参不受影响。"""
    import logging

    long_arg = "短前缀" + "z" * 300
    received = {}

    def record(city, **kw):
        received["city"] = city
        received["kw"] = kw
        return "晴"

    with caplog.at_level(logging.INFO, logger="agent"):
        out = dt._safe_call("t", record, long_arg, extra="api_key=sk-OKLOG-998877665544")

    assert out == "晴", "正常调用返回值不变"
    assert received["city"] == long_arg, "实际传参不得被清洗影响"
    assert received["kw"] == {"extra": "api_key=sk-OKLOG-998877665544"}

    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "diagnostic_tool_success"
    ]
    assert entries
    entry = entries[-1]
    serialized = str(entry)
    assert "sk-OKLOG-998877665544" not in serialized, "成功日志也不得含原始密钥"
    assert "z" * 200 not in serialized, "成功日志不得完整记录超长参数"
    assert entry["args"][0].startswith("短前缀"), "清洗后保留参数前缀"


def test_conversation_safe_call_semantics_match_diagnostic(caplog):
    """对话工具模块与诊断工具模块的 _safe_call 保持相同安全语义。"""
    import logging

    from agent.tools import agent_tools as at

    with caplog.at_level(logging.INFO, logger="agent"):
        out = at._safe_call("t", lambda q, **kw: "结果", "查询", mode="api_key=sk-CONV-998877665544")

    assert out == "结果"
    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "tool_success"
    ]
    assert entries
    assert "sk-CONV-998877665544" not in str(entries[-1])


# ----------------------------------------------------------------- 中间件日志清洗（P1-10）
class _StubRuntime:
    def __init__(self):
        self.context = {}


class _StubToolRequest:
    def __init__(self, name, args):
        self.tool_call = {"name": name, "args": args}
        self.runtime = _StubRuntime()


def _formatter_outputs(caplog):
    """把 caplog 捕获的记录用生产 formatter 格式化，返回 (json_outputs, console_outputs)。"""
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    json_outputs = [JsonFormatter().format(r) for r in caplog.records]
    console_outputs = [CONSOLE_FORMAT.format(r) for r in caplog.records]
    return json_outputs, console_outputs


def test_middleware_tool_monitor_logs_sanitized(caplog):
    """monitor_tool 中间件：工具名与参数含密钥 / 路径 / 长文本时，formatter 输出零泄漏。"""
    import logging

    from agent.tools.middleware import monitor_tool

    secret = "sk-MIDDLE-998877665544"
    abs_path = "D:\\secret\\tools\\middleware.bin"
    long_text = "超长参数" * 60

    request = _StubToolRequest("get_weather", {"city": f"{secret} {abs_path} {long_text}"})

    def handler(req):
        raise RuntimeError(f"调用失败 api_key={secret} 路径 {abs_path}")

    with caplog.at_level(logging.INFO, logger="agent"), pytest.raises(RuntimeError):
        # @wrap_tool_call 装饰器把函数绑定为中间件实例的 wrap_tool_call 方法
        monitor_tool.wrap_tool_call(request, handler)

    json_outs, console_outs = _formatter_outputs(caplog)
    assert json_outs and console_outs, "应有日志记录"

    events = [r.msg["event"] for r in caplog.records if isinstance(r.msg, dict)]
    assert "tool_monitor_start" in events, "结构化事件名保留"
    assert "tool_monitor_error" in events

    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：原始密钥不得泄漏"
        assert abs_path not in combined, f"{label}：绝对路径不得泄漏"
        assert abs_path.replace("\\", "\\\\") not in combined, f"{label}：转义路径不得泄漏"
        assert long_text not in combined, f"{label}：超长文本不得完整出现"
        assert "***REDACTED***" in combined or "<PATH_REDACTED>" in combined, f"{label}：应有脱敏标记"


def test_middleware_before_model_logs_sanitized(caplog):
    """log_before_model：用户消息含密钥 / 路径 / 长文本时，formatter 输出零泄漏。"""
    import logging

    from langchain_core.messages import HumanMessage

    from agent.tools.middleware import log_before_model

    secret = "sk-BEFORE-998877665544"
    abs_path = "D:\\secret\\prompts\\system.bin"
    long_text = "用户超长输入" * 60

    state = {"messages": [HumanMessage(f"{secret} {abs_path} {long_text}")]}

    with caplog.at_level(logging.DEBUG, logger="agent"):
        # @before_model 装饰器把函数绑定为中间件实例的 before_model 方法
        assert log_before_model.before_model(state, None) is None

    json_outs, console_outs = _formatter_outputs(caplog)
    assert json_outs, "应有日志记录"

    info_events = [r.msg for r in caplog.records
                   if isinstance(r.msg, dict) and r.msg.get("event") == "before_model_call"]
    assert info_events, "before_model_call 事件保留"
    assert info_events[0]["message_count"] == 1, "消息计数保留（安全字段）"

    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：原始密钥不得泄漏"
        assert abs_path not in combined, f"{label}：绝对路径不得泄漏"
        assert long_text not in combined, f"{label}：超长文本不得完整出现"


def test_mock_weather_service_city_param_sanitized(caplog):
    """Mock 服务工具参数：city 含密钥 / 路径 / 长文本时，formatter 输出零泄漏且返回值不变。"""
    import logging

    from agent.services.mock_services import MockWeatherService

    secret = "sk-CITY-998877665544"
    abs_path = "D:\\secret\\weather\\cache.bin"
    long_text = "城市名超长输入" * 60
    city = f"{secret} {abs_path} {long_text}"

    with caplog.at_level(logging.INFO, logger="agent"):
        result = MockWeatherService().get_weather(city)

    assert city in result, "返回值（业务行为）不受日志清洗影响"
    assert "天气为晴天" in result

    json_outs, console_outs = _formatter_outputs(caplog)
    events = [r.msg["event"] for r in caplog.records if isinstance(r.msg, dict)]
    assert "mock_weather" in events, "事件名保留"

    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：工具参数中的密钥不得泄漏"
        assert abs_path not in combined, f"{label}：工具参数中的路径不得泄漏"
        assert long_text not in combined, f"{label}：超长参数不得完整出现"


# ----------------------------------------------------------------- 工具返回值安全边界（P1-11）
def test_http_weather_service_error_returns_fixed_message(monkeypatch):
    """HTTP 天气服务异常：返回固定安全文案，不回显 city / 异常细节 / 远端响应。"""
    import requests

    from agent.services.http_services import HttpWeatherService

    secret = "sk-HTTP-998877665544"
    abs_path = "D:\\secret\\http\\weather.bin"
    remote_body = '{"internal": "远端服务原始响应内容"}'
    city = "杭州" + "A" * 120  # 用户/模型可控的入参（超长）

    def boom(*a, **kw):
        raise requests.ConnectionError(f"连接失败 api_key={secret} {abs_path} resp={remote_body}")

    monkeypatch.setattr("agent.services.http_services.requests.get", boom)

    result = HttpWeatherService("http://weather.internal/api").get_weather(city)

    assert result == "获取天气信息失败，请稍后重试", "失败返回必须是固定安全文案"
    for leaked in (secret, abs_path, remote_body, "ConnectionError", city, "杭州"):
        assert leaked not in result, f"失败返回不得包含：{leaked[:30]}"


def test_no_exception_text_concatenated_into_tool_returns_static_scan():
    """静态扫描：生产代码任何 return 语句不得拼接 str(e)/异常变量/traceback。

    用 AST 提取完整 return 表达式（覆盖跨行 f-string），杜绝
    「return f\"工具{tool}调用失败：{str(e)}\"」形态回流。
    logger_handler 的脱敏 helper（safe_exception_fields / log_safe_text 等）
    是唯一合法入口：异常文本经清洗后才返回，扫描时排除其内部 return。
    """
    import ast
    import re
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    dangerous = re.compile(
        r"str\(\s*(?:e|exc|error|err)\s*\)"
        r"|\{\s*(?:str\(\s*)?(?:e|exc|error|err)\s*[!}:]"
        r"|type\(\s*(?:e|exc|error)\s*\)"
        r"|traceback\.format_exc\(\)"
    )
    sanitizer_funcs = {
        "safe_exception_fields", "log_safe_text", "log_safe_value",
        "_redact", "_redact_keys",
    }

    violations: list[str] = []
    for pkg in ("agent", "api", "rag", "utils", "mcp_server"):
        for path in (project_root / pkg).rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
            exempt: set[int] = set()
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef) and fn.name in sanitizer_funcs:
                    for node in ast.walk(fn):
                        if isinstance(node, ast.Return):
                            exempt.add(id(node))
            for node in ast.walk(tree):
                if isinstance(node, ast.Return) and node.value is not None and id(node) not in exempt:
                    segment = ast.get_source_segment(src, node) or ""
                    if dangerous.search(segment):
                        violations.append(f"{path.relative_to(project_root)}:{node.lineno}")

    assert not violations, (
        f"以下 return 语句把异常文本拼进了工具/服务返回值（客户端或模型可见）：\n{violations}"
    )
