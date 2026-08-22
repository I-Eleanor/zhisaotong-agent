"""工具类测试：file_handler 修复、config_handler、path_tool、logger、模型工厂、prompt 加载。"""
import os

from utils import file_handler, path_tool
from utils.config_handler import agent_conf, chroma_conf, prompts_conf, rag_conf


def test_listdir_returns_empty_for_nondir():
    result = file_handler.listdir_with_allowed_type("/不存在的路径/xyz", ("txt", "pdf"))
    assert result == [], f"非目录应返回 []，实际返回 {result!r}"


def test_listdir_filters_by_extension(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "c.md").write_text("x", encoding="utf-8")
    found = file_handler.listdir_with_allowed_type(str(tmp_path), ("txt", "pdf"))
    names = {os.path.basename(p) for p in found}
    assert names == {"a.txt", "b.pdf"}


def test_md5_deterministic_and_distinct(tmp_path):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("hello", encoding="utf-8")
    f2.write_text("world", encoding="utf-8")
    h1 = file_handler.get_file_md5_hex(str(f1))
    h2 = file_handler.get_file_md5_hex(str(f2))
    assert h1 == file_handler.get_file_md5_hex(str(f1)), "同一文件 MD5 必须一致"
    assert h1 != h2, "不同文件 MD5 必须不同"
    assert isinstance(h1, str) and len(h1) == 32


def test_md5_missing_file_returns_none(tmp_path):
    assert file_handler.get_file_md5_hex(str(tmp_path / "nope.txt")) is None


def test_config_handler_loads_all():
    assert isinstance(rag_conf, dict) and rag_conf.get("chat_model_name")
    assert isinstance(chroma_conf, dict) and chroma_conf.get("collection_name")
    assert isinstance(agent_conf, dict)
    assert isinstance(prompts_conf, dict)


def test_path_tool_resolves_relative():
    abs_path = path_tool.get_abs_path("config/rag.yml")
    assert os.path.isabs(abs_path)
    assert abs_path.endswith("config/rag.yml") or abs_path.endswith("config\\rag.yml")


def test_logger_handler_emits():
    from utils.logger_handler import logger
    logger.info({"event": "test_log", "value": 1})


# ----------------------------------------------------------------- 服务异常边界（P0 任务 3.1）
def test_service_unavailable_error_is_project_error():
    """ServiceUnavailableError 继承 AgentProjectError，携带稳定元数据。"""
    from utils import error_codes
    from utils.exceptions import AgentProjectError, ServiceUnavailableError

    assert issubclass(ServiceUnavailableError, AgentProjectError)
    exc = ServiceUnavailableError("未查询到用户的设备运行数据")
    assert exc.stage == "service"
    assert exc.retryable is True
    assert exc.error_code == error_codes.SERVICE_UNAVAILABLE


def test_interfaces_no_longer_defines_business_exceptions():
    """interfaces.py 只保留服务接口（ABC），业务异常统一在 utils/exceptions.py。"""
    import inspect

    import agent.services.interfaces as interfaces
    import utils.exceptions as exceptions

    for name, obj in inspect.getmembers(interfaces, inspect.isclass):
        assert not issubclass(obj, Exception) or obj.__module__ == "builtins", (
            f"interfaces.py 不得定义业务异常，发现 {name}（{obj.__module__}）"
        )
    assert hasattr(exceptions, "ServiceUnavailableError"), "异常应定义在 utils/exceptions.py"


# ----------------------------------------------------------------- 日志脱敏（P2 代码质量）
def test_log_safe_text_truncates_long_input():
    """超长用户输入进日志时被截断，不完整记录。"""
    from utils.logger_handler import log_safe_text

    long_query = "我的扫地机坏了，" + "x" * 300
    result = log_safe_text(long_query)
    assert len(result) < 120, "超长输入应被截断"
    assert "x" * 200 not in result, "不得完整记录超长输入"


def test_log_safe_text_collapses_whitespace_and_handles_none():
    """换行压缩为空格；None/空串安全处理。"""
    from utils.logger_handler import log_safe_text

    assert log_safe_text("第一行\n第二行") == "第一行 第二行"
    assert log_safe_text(None) == ""
    assert log_safe_text("") == ""


def test_log_safe_text_keeps_short_input():
    """短输入保持原样，便于日志定位问题。"""
    from utils.logger_handler import log_safe_text

    assert log_safe_text("扫地机不工作") == "扫地机不工作"


# ----------------------------------------------------------------- 统一异常摘要（P1-6 异常日志收敛）
def test_safe_exception_fields_exact_fields():
    """safe_exception_fields 固定返回 error_type + error_msg 两个键，不多不少。"""
    from utils.logger_handler import safe_exception_fields

    exc = RuntimeError("boom")
    fields = safe_exception_fields(exc)

    assert fields == {
        "error_type": "RuntimeError",
        "error_msg": "boom",
    }, "异常摘要字段必须精确为 error_type / error_msg"


def test_safe_exception_fields_collapses_multiline_and_truncates():
    """多行异常消息（含异常链文本）被压成单行，超长被截断。"""
    from utils.logger_handler import LOG_TEXT_MAX_LENGTH, safe_exception_fields

    long_head = "首行错误详情" + "x" * 150
    exc = ValueError(f"{long_head}\nThe above exception was the direct cause of...")
    fields = safe_exception_fields(exc)

    assert "\n" not in fields["error_msg"], "异常摘要必须单行化"
    assert len(fields["error_msg"]) <= LOG_TEXT_MAX_LENGTH + 20  # 截断后缀余量
    assert "direct cause" not in fields["error_msg"], "超长异常链文本只保留前缀"


def test_safe_exception_fields_redacts_prefixed_and_bare_keys():
    """带前缀密钥与裸 sk- 密钥在异常摘要中均被脱敏。"""
    from utils.logger_handler import safe_exception_fields

    exc = RuntimeError("连接失败 api_key=sk-PFX-112233445566 与裸密钥 sk-BARE-998877665544")
    fields = safe_exception_fields(exc)

    assert "sk-PFX-112233445566" not in fields["error_msg"]
    assert "sk-BARE-998877665544" not in fields["error_msg"]
    assert "***REDACTED***" in fields["error_msg"]


def test_redact_bare_sk_key_requires_min_length():
    """裸密钥脱敏：sk- 后至少 8 位字符才匹配，普通短文本不受影响。"""
    from utils.logger_handler import log_safe_text

    assert "sk-1234567890abc" not in log_safe_text("密钥 sk-1234567890abc 泄漏")
    assert "sk-abcdefgh" not in log_safe_text("密钥 sk-abcdefgh 泄漏"), "8 位应被脱敏"
    # sk- 后不足 8 位：普通短文本，保持原样
    assert "sk-abc" in log_safe_text("短文本 sk-abc 不脱敏")


def test_redact_bare_sk_key_word_boundary_avoids_false_positive():
    """\\b 边界：task-management / disk-usage 等普通词不被误伤。"""
    from utils.logger_handler import log_safe_text

    text = "task-management disk-usage risk-assessment 正常词汇"
    assert log_safe_text(text) == text, "不含裸密钥的普通文本不得被误改写"


# ----------------------------------------------------------------- 路径与隐私文本脱敏（P1-9）
def test_redact_windows_and_unix_absolute_paths():
    """Windows 盘符路径与 Unix 多段路径统一替换为 <PATH_REDACTED>。"""
    from utils.logger_handler import log_safe_text

    out = log_safe_text("读取 D:\\secret\\internal\\planner_cache.bin 失败")
    assert "<PATH_REDACTED>" in out
    assert "D:\\secret" not in out and "planner_cache" not in out

    out = log_safe_text("配置 C:\\Users\\ops\\.dashscope\\config.json 缺失")
    assert "<PATH_REDACTED>" in out
    assert "C:\\Users" not in out

    out = log_safe_text("写入 /var/log/app/server.log 失败")
    assert "<PATH_REDACTED>" in out
    assert "/var/log" not in out and "server.log" not in out

    out = log_safe_text("读取 /home/user/data/config.json 出错")
    assert "<PATH_REDACTED>" in out
    assert "/home/user" not in out

    # 中文紧邻路径（无空格分隔）同样识别
    out = log_safe_text("路径/home/user/data/config.json泄漏")
    assert "<PATH_REDACTED>" in out
    assert "/home/user" not in out


def test_redact_path_containing_secret():
    """路径中含密钥：密钥先脱敏、路径再整体替换，两者都不泄漏。"""
    from utils.logger_handler import log_safe_text

    out = log_safe_text("D:\\keys\\api_key=sk-INPATH-112233445566.bin 加载失败")
    assert "D:\\keys" not in out, "路径前缀不得泄漏"
    assert "sk-INPATH-112233445566" not in out, "路径中的密钥不得泄漏"
    assert "<PATH_REDACTED>" in out
    assert "***REDACTED***" in out


def test_redact_path_and_secret_together():
    """路径与密钥同时出现时均脱敏，互不干扰。"""
    from utils.logger_handler import log_safe_text

    secret = "sk-MIXED-998877665544"
    out = log_safe_text(f"读取 D:\\app\\config 失败，密钥 {secret} 泄漏")
    assert "<PATH_REDACTED>" in out
    assert "***REDACTED***" in out
    assert "D:\\app" not in out
    assert secret not in out


def test_redact_keeps_relative_paths_urls_and_chinese():
    """相对路径 / URL / 中文文本不被误伤；URL 内密钥仍脱敏。"""
    from utils.logger_handler import log_safe_text

    # 普通 URL 不被误判为 Unix 路径
    url = "https://api.example.com/v1/data/list"
    assert url in log_safe_text(f"请求 {url} 超时")

    # URL 查询串中的密钥脱敏，但 URL 结构保留
    out = log_safe_text("回调 https://api.example.com/v1/callback?token=abcdefgh123456 失败")
    assert "https://api.example.com/v1/callback" in out
    assert "token=***REDACTED***" in out
    assert "abcdefgh123456" not in out

    # 短相对路径与模块相对路径不被过度替换
    for rel in ("data/file.txt", "./config", "../config/rag.yml", "tests/test_utils.py",
                "agent/diagnostic/nodes.py", "chroma_db/store"):
        assert rel in log_safe_text(f"加载 {rel} 完成"), f"相对路径 {rel} 不应被替换"

    # 时间 / 日期 / 分数等含冒号斜杠的普通文本不受影响
    for plain in ("12:30:45", "24/08/22", "10/3", "key: value"):
        assert plain in log_safe_text(f"时间 {plain} 记录")

    # 正常中文文本原样保留
    text = "扫地机不工作，请检查滤网与电池触点"
    assert log_safe_text(text) == text


def test_log_safe_value_cleans_nested_paths():
    """log_safe_value 递归沿用同一脱敏规则：嵌套结构内的路径被替换。"""
    from utils.logger_handler import log_safe_value

    value = {
        "config_path": "D:\\secrets\\gateway\\token.bin",
        "paths": ["/var/log/agent/app.log", "/etc/ssl/private.key"],
        "nested": {"model_dir": "/home/ops/models/qwen"},
    }
    out = log_safe_value(value)
    serialized = str(out)

    assert "D:\\secrets" not in serialized
    assert "/var/log" not in serialized
    assert "/etc/ssl" not in serialized
    assert "/home/ops" not in serialized
    assert serialized.count("<PATH_REDACTED>") == 4, "四处路径全部替换"
    assert isinstance(out["paths"], list), "结构形态保留"


def test_formatters_redact_paths_consistently():
    """控制台与 JSON formatter 输出一致安全：路径（含 JSON 转义形态）均替换。"""
    import logging

    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    secret_path = "D:\\secret\\internal\\planner_cache.bin"
    record = logging.LogRecord(
        name="agent",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg={
            "event": "knowledge_load_failed",
            "file": secret_path,
            "error_msg": f"读取 {secret_path} 失败 api_key=sk-PATHFMT-998877665544",
        },
        args=(),
        exc_info=None,
        func="test",
    )

    json_out = JsonFormatter().format(record)
    # json.dumps 会把反斜杠转义为 \\：转义形态的路径同样不得泄漏
    assert secret_path not in json_out
    assert secret_path.replace("\\", "\\\\") not in json_out, "JSON 转义形态的路径也不得泄漏"
    assert "<PATH_REDACTED>" in json_out
    assert "sk-PATHFMT-998877665544" not in json_out

    console_out = CONSOLE_FORMAT.format(record)
    assert secret_path not in console_out
    assert secret_path.replace("\\", "\\\\") not in console_out, "repr 转义形态的路径也不得泄漏"
    assert "<PATH_REDACTED>" in console_out
    assert "sk-PATHFMT-998877665544" not in console_out


# ----------------------------------------------------------------- 受信任路由字段保护（P1-9.1）
def _route_log_record(msg):
    import logging

    return logging.LogRecord(
        name="agent",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
        func="test",
    )


def test_trusted_route_path_preserved_in_json_and_console():
    """/api/chat/sync 等应用路由在 JSON 与控制台日志中原样保留。"""
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    for route in ("/api/chat/sync", "/api/diagnose", "/api/knowledge/rebuild",
                  "/api/health/ready", "/health/ready", "/api", "/health"):
        record = _route_log_record({"event": "unhandled_project_error", "path": route, "stage": "sync_chat"})
        json_out = JsonFormatter().format(record)
        console_out = CONSOLE_FORMAT.format(record)
        assert f'"{route}"' in json_out, f"JSON 日志应保留应用路由 {route}"
        assert f"'{route}'" in console_out, f"控制台日志应保留应用路由 {route}"
        assert "__TRUSTED_ROUTE__" not in json_out, "占位符不得残留在输出中"
        assert "__TRUSTED_ROUTE__" not in console_out


def test_malicious_path_field_values_still_redacted():
    """path 字段被写入恶意绝对路径 / 目录穿越串时仍被替换，不能借 path 字段绕过脱敏。"""
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    for malicious in (
        "D:\\secret\\internal\\key.bin",
        "/home/user/secret/data.bin",
        "/api/../secret/key.bin",
        "/api/chat/../../etc/passwd",
    ):
        record = _route_log_record({"event": "auth_failed", "path": malicious})
        json_out = JsonFormatter().format(record)
        console_out = CONSOLE_FORMAT.format(record)
        for label, out in (("JSON", json_out), ("控制台", console_out)):
            assert "<PATH_REDACTED>" in out, f"{label}：恶意路径必须替换：{malicious}"
            assert "secret" not in out, f"{label}：路径细节不得泄漏：{malicious}"
            assert "passwd" not in out


def test_trusted_route_kept_while_secrets_and_file_field_redacted():
    """路由保留的同时：密钥脱敏行为不变，file 字段的本地路径仍按现有规则替换。"""
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    local_path = "D:\\secret\\models\\weights.bin"
    record = _route_log_record({
        "event": "unhandled_project_error",
        "path": "/api/chat/sync",
        "stage": "sync_chat",
        "file": local_path,
        "error_msg": "加载失败 api_key=sk-ROUTE-998877665544",
    })

    json_out = JsonFormatter().format(record)
    console_out = CONSOLE_FORMAT.format(record)
    for label, out in (("JSON", json_out), ("控制台", console_out)):
        assert "/api/chat/sync" in out, f"{label}：应用路由保留"
        assert "sk-ROUTE-998877665544" not in out, f"{label}：密钥脱敏行为不变"
        assert "api_key=***REDACTED***" in out, f"{label}：密钥以脱敏标记呈现"
        assert "<PATH_REDACTED>" in out, f"{label}：file 字段的本地路径仍被替换"
        assert "secret" not in out, f"{label}：本地源码路径不得恢复"


def test_trusted_route_protection_copies_dict_without_mutation():
    """保护逻辑复制日志字典：原 record.msg 不被修改，非字典消息不受影响。"""
    from utils.logger_handler import _protect_trusted_route

    msg = {"event": "e", "path": "/api/chat/sync", "query": "用户问题"}
    patched, route = _protect_trusted_route(msg)
    assert route == "/api/chat/sync"
    assert patched["path"] == "__TRUSTED_ROUTE__"
    assert msg["path"] == "/api/chat/sync", "原字典不得被修改"

    # 非路由值 / 非字符串 / 非字典：一律不保护
    for value in ("/home/user/data", "D:\\x\\y.bin", 123, None, "chat/sync"):
        patched, route = _protect_trusted_route({"event": "e", "path": value})
        assert route is None, f"{value!r} 不应被视为受信任路由"
        assert patched is not None
    patched, route = _protect_trusted_route("普通字符串消息")
    assert route is None and patched == "普通字符串消息"


# ----------------------------------------------------------------- 递归日志清洗（P1-7 工具边界）
def test_log_safe_value_cleans_nested_containers_recursively():
    """嵌套 dict / list / tuple 递归清洗：字符串走 log_safe_text，结构保留。"""
    from utils.logger_handler import log_safe_value

    value = {
        "query": "api_key=sk-NESTED-112233445566 的查询",
        "items": ["sk-PLAIN-998877665544", 42, None, True],
        "nested": {"inner": ("token=TOKEN-INNER-99887766", 3.14)},
    }
    out = log_safe_value(value)
    serialized = str(out)

    assert "sk-NESTED-112233445566" not in serialized
    assert "sk-PLAIN-998877665544" not in serialized
    assert "TOKEN-INNER-99887766" not in serialized
    assert out["items"][1:] == [42, None, True], "标量原样保留"
    assert out["nested"]["inner"][1] == 3.14
    assert isinstance(out["items"], list), "tuple/list 统一为清洗后的列表形态"


def test_log_safe_value_truncates_oversized_collection():
    """超长集合截断：只保留前 8 个元素并标记截断数量。"""
    from utils.logger_handler import LOG_VALUE_MAX_ITEMS, log_safe_value

    big_list = [f"item{i}" for i in range(20)]
    out = log_safe_value(big_list)
    assert isinstance(out, list)
    assert len(out) == LOG_VALUE_MAX_ITEMS + 1, "8 个元素 + 1 个截断标记"
    assert "<truncated 12 more>" in out

    big_dict = {f"k{i}": i for i in range(10)}
    out_d = log_safe_value(big_dict)
    assert isinstance(out_d, dict)
    assert len(out_d) == LOG_VALUE_MAX_ITEMS + 1
    assert out_d["<truncated>"] == "2 more"


def test_log_safe_value_depth_limit_prevents_recursion_blowup():
    """超深嵌套返回占位符；循环引用结构也不会递归爆栈。"""
    import sys

    from utils.logger_handler import log_safe_value

    deep: list = []
    node = deep
    for _ in range(50):
        child: list = []
        node.append(child)
        node = child

    out = log_safe_value(deep)  # 50 层嵌套：不抛 RecursionError
    assert "<max_depth>" in str(out)

    # 自引用循环：深度上限天然终止递归
    loop: list = []
    loop.append(loop)
    out_loop = log_safe_value(loop)
    assert "<max_depth>" in str(out_loop)
    assert sys.getrefcount(loop) > 0, "原结构未被破坏"


def test_log_safe_value_does_not_mutate_original():
    """只清洗日志副本：原参数（含嵌套内容与超长字符串）保持不变。"""
    from utils.logger_handler import log_safe_value

    original = {
        "secret": "api_key=sk-ORIG-112233445566",
        "long": "x" * 300,
        "items": [{"a": 1}, {"b": "sk-ORIG-998877665544"}],
    }
    snapshot = {
        "secret": original["secret"],
        "long": original["long"],
        "items": [{"a": 1}, {"b": "sk-ORIG-998877665544"}],
    }
    _ = log_safe_value(original)

    assert original == snapshot, "原参数不得被修改"
    assert original["long"] == "x" * 300, "超长字符串原样保留在原参数中"


def test_log_safe_value_passes_short_plain_scalars():
    """短标量与普通短字符串正常通过，便于日志定位。"""
    from utils.logger_handler import log_safe_value

    assert log_safe_value("普通查询") == "普通查询"
    assert log_safe_value(7) == 7
    assert log_safe_value(None) is None
    assert log_safe_value(True) is True
    assert log_safe_value(["a", "b"]) == ["a", "b"]


# ----------------------------------------------------------------- 模型工厂懒加载与缓存
def test_model_factory_lazy_and_cache():
    import model.factory as mf
    mf.reset_models()
    assert mf._chat_model is None, "初始应为 None（懒加载）"
    assert mf._embed_model is None


def test_model_factory_reset():
    import model.factory as mf
    mf.reset_models()
    assert mf._chat_model is None
    assert mf._embed_model is None


# ----------------------------------------------------------------- Prompt 缺失行为
def test_prompt_missing_key_raises():
    from utils.prompt_loader import _cache, _load_prompt
    _cache.pop("__nonexistent_key__", None)
    import pytest
    with pytest.raises(KeyError):
        _load_prompt("__nonexistent_key__", "测试")


# ----------------------------------------------------------------- 文件格式白名单
def test_file_whitelist_rejects_exe(tmp_path):
    (tmp_path / "good.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "bad.exe").write_text("nope", encoding="utf-8")
    found = file_handler.listdir_with_allowed_type(str(tmp_path), ("txt", "pdf"))
    names = {os.path.basename(p) for p in found}
    assert "good.txt" in names
    assert "bad.exe" not in names


# ----------------------------------------------------------------- 重试语义（P1 代码质量 / 11.1 恢复普通异常重试）
def test_retry_retries_unexpected_exceptions_with_backoff():
    """普通异常按指数退避重试：max_retries=2 时共执行 3 次，最终抛出原异常。"""
    from utils.resilience import retry_with_backoff

    calls = []

    def bad_func():
        calls.append(1)
        raise ValueError("参数错误：query 不能为空")

    import pytest
    with pytest.raises(ValueError, match="参数错误：query 不能为空") as exc_info:
        retry_with_backoff(bad_func, max_retries=2, base_delay=0.01, label="t")

    assert len(calls) == 3, "初始 1 次 + 重试 2 次"
    assert type(exc_info.value) is ValueError, "最终异常保持原始类型"
    assert str(exc_info.value) == "参数错误：query 不能为空", "最终异常保持原始消息"


def test_retry_unexpected_logs_sanitized(caplog):
    """普通异常重试日志（retry_attempt_unexpected / retry_exhausted）不泄漏密钥与路径。"""
    import logging

    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter
    from utils.resilience import retry_with_backoff

    secret = "sk-RETRY-998877665544"
    abs_path = "D:\\secret\\retry\\model.bin"

    def always_fails():
        raise RuntimeError(f"连接失败 api_key={secret} {abs_path}")

    import pytest
    with caplog.at_level(logging.WARNING, logger="agent"), pytest.raises(RuntimeError):
        retry_with_backoff(always_fails, max_retries=2, base_delay=0.01, label="t")

    events = [r.msg.get("event") for r in caplog.records if isinstance(r.msg, dict)]
    assert events.count("retry_attempt_unexpected") == 2, "前两次失败各记一次重试日志"
    assert "retry_exhausted" in events, "耗尽后记录 retry_exhausted"

    json_outs = [JsonFormatter().format(r) for r in caplog.records]
    console_outs = [CONSOLE_FORMAT.format(r) for r in caplog.records]
    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：原始密钥不得泄漏"
        assert abs_path not in combined, f"{label}：绝对路径不得泄漏"
        assert abs_path.replace("\\", "\\\\") not in combined, f"{label}：转义路径不得泄漏"


def test_retry_does_not_retry_non_retryable_project_errors():
    """retryable=False 的项目异常（确定性失败）不得重试。"""
    from utils.exceptions import ToolExecutionError
    from utils.resilience import retry_with_backoff

    calls = []

    def failing():
        calls.append(1)
        raise ToolExecutionError("工具确定性失败")

    import pytest
    with pytest.raises(ToolExecutionError):
        retry_with_backoff(failing, max_retries=3, base_delay=0.01, label="t")

    assert len(calls) == 1, "不可重试错误应立即抛出"


def test_retry_retries_retryable_project_errors():
    """retryable=True 的项目异常应按指数退避重试并最终成功。"""
    from utils.exceptions import ModelInvocationError
    from utils.resilience import retry_with_backoff

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ModelInvocationError("模型暂时不可用")
        return "ok"

    result = retry_with_backoff(flaky, max_retries=3, base_delay=0.01, label="t")
    assert result == "ok"
    assert len(calls) == 3, "可重试错误应重试到成功"


def test_retry_exhausted_raises():
    """可重试错误超过最大次数后抛出。"""
    from utils.exceptions import RetrievalError
    from utils.resilience import retry_with_backoff

    calls = []

    def always_fails():
        calls.append(1)
        raise RetrievalError("检索持续失败")

    import pytest
    with pytest.raises(RetrievalError):
        retry_with_backoff(always_fails, max_retries=2, base_delay=0.01, label="t")
    assert len(calls) == 3, "初始 1 次 + 重试 2 次"


# ----------------------------------------------------------------- 逐字输出延迟可关闭（P1 代码质量）
def test_stream_chunk_delay_disabled(monkeypatch):
    """STREAM_CHUNK_DELAY_SECONDS=0 时关闭逐字延迟，不占用工作线程等待。"""
    import time as time_mod

    from langchain_core.messages import AIMessageChunk

    import agent.conversation_agent as ca
    from agent.conversation_agent import ConversationAgent

    class _StubAgent:
        def stream(self, input_dict, stream_mode=None, context=None):
            yield AIMessageChunk(content="这是二十个字符的回答内容用于测试！"), {}

    agent = ConversationAgent.__new__(ConversationAgent)
    agent.agent = _StubAgent()

    monkeypatch.setattr(ca, "STREAM_CHUNK_DELAY_SECONDS", 0.0)

    start = time_mod.perf_counter()
    events = list(agent.stream("你好", []))
    elapsed = time_mod.perf_counter() - start

    messages = [e["content"] for e in events if e["type"] == "message"]
    assert "".join(messages) == "这是二十个字符的回答内容用于测试！"
    assert elapsed < 0.1, (
        f"延迟关闭后流式输出耗时 {elapsed:.3f}s，不应存在逐字 sleep 占用线程"
    )
    assert events[-1]["type"] == "done"


# ----------------------------------------------------------------- 服务层日志清洗（P1-10）
def test_mcp_fallback_and_file_handler_logs_sanitized(caplog, tmp_path):
    """resilience 与 file_handler：异常 / 路径含密钥与绝对路径时，formatter 输出零泄漏。"""
    import logging

    from utils import file_handler
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter
    from utils.resilience import mcp_fallback_response

    secret = "sk-SVC-998877665544"
    abs_path = "D:\\secret\\services\\data.bin"
    long_text = "服务错误详情" * 60

    with caplog.at_level(logging.WARNING, logger="agent"):
        response = mcp_fallback_response("get_device_status", RuntimeError(f"连接失败 api_key={secret} {abs_path} {long_text}"))
        assert "不可用" in response, "降级响应行为不变"

        # file_handler：不存在路径（含密钥形态）触发 md5 错误日志
        assert file_handler.get_file_md5_hex(f"{abs_path}") is None

    json_outs = [JsonFormatter().format(r) for r in caplog.records]
    console_outs = [CONSOLE_FORMAT.format(r) for r in caplog.records]
    assert json_outs, "应有日志记录"

    events = {r.msg.get("event") for r in caplog.records if isinstance(r.msg, dict)}
    assert "mcp_fallback" in events, "mcp_fallback 事件名保留"
    assert "md5_file_not_found" in events, "md5 错误事件保留"

    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：原始密钥不得泄漏"
        assert abs_path not in combined, f"{label}：绝对路径不得泄漏"
        assert abs_path.replace("\\", "\\\\") not in combined, f"{label}：转义路径不得泄漏"
        assert long_text not in combined, f"{label}：超长异常文本不得完整出现"
        assert "traceback" not in combined.lower() or "Traceback" not in combined


# ----------------------------------------------------------------- 配置校验安全（P1-13）
def test_validate_model_config_rejects_invalid_timeout(monkeypatch):
    """非法超时值（负数/零/非数值/布尔）被拒绝，缺省与正数合法。"""
    from utils.config_handler import rag_conf
    from utils.config_validator import validate_model_config

    for bad in (-1, 0, "abc", True):
        monkeypatch.setattr(
            "utils.config_validator.rag_conf",
            {**rag_conf, "llm_timeout_seconds": bad},
        )
        errors = validate_model_config()
        assert any(e[0] == "llm_timeout_seconds" for e in errors), f"非法超时值 {bad!r} 应被拒绝"
        # 错误信息不含原值（原值可能携带任意文本）
        assert all("abc" not in e[1] for e in errors if e[0] == "llm_timeout_seconds")

    for good in (30, 60, 1.5):
        monkeypatch.setattr(
            "utils.config_validator.rag_conf",
            {**rag_conf, "llm_timeout_seconds": good},
        )
        assert not any(
            e[0] == "llm_timeout_seconds" for e in validate_model_config()
        ), f"合法超时值 {good!r} 不应报错"


def test_validate_model_config_rejects_invalid_model_names(monkeypatch):
    """非法模型名被拒绝：不支持的 embedding 模式、含空白/路径分隔符的 chat 模型名。"""
    from utils.config_handler import rag_conf
    from utils.config_validator import validate_model_config

    monkeypatch.setattr(
        "utils.config_validator.rag_conf",
        {**rag_conf, "embedding_model_name": "gpt-4-unknown"},
    )
    errors = validate_model_config()
    assert any(e[0] == "embedding_model_name" for e in errors), "不支持的 embedding 模式应被拒绝"
    assert all("gpt-4-unknown" not in e[1] for e in errors), "错误信息不含原值"

    for bad_chat in ("deepseek chat", "a/b", "a\\b"):
        monkeypatch.setattr(
            "utils.config_validator.rag_conf",
            {**rag_conf, "chat_model_name": bad_chat},
        )
        errors = validate_model_config()
        assert any(e[0] == "chat_model_name" for e in errors), f"非法模型名 {bad_chat!r} 应被拒绝"


def test_validate_cors_origins_format(monkeypatch):
    """CORS 格式校验：通配符与显式来源合法，非法格式被拒绝。"""
    from utils.config_validator import validate_cors_origins

    monkeypatch.setenv("CORS_ORIGINS", "*")
    assert validate_cors_origins() == [], "通配符本身合法（警告由应用层记录）"

    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com, http://localhost:3000")
    assert validate_cors_origins() == [], "显式来源列表合法"

    monkeypatch.setenv("CORS_ORIGINS", "app.example.com, javascript:alert(1)")
    errors = validate_cors_origins()
    assert len(errors) == 2, "裸域名与 javascript: 伪协议均应被拒绝"
    assert all(e[0] == "CORS_ORIGINS" for e in errors)

    monkeypatch.setenv("CORS_ORIGINS", "")
    assert validate_cors_origins() == [], "空值合法（等价于禁止跨域）"


def test_validate_errors_do_not_leak_full_local_paths(monkeypatch, tmp_path):
    """配置校验错误只含文件名，不含完整本地绝对路径。"""
    from utils.config_handler import rag_conf
    from utils.config_validator import ConfigValidationError, validate_before_use, validate_paths

    missing = tmp_path / "secret" / "models" / "nope.bin"
    monkeypatch.setattr(
        "utils.config_validator.rag_conf",
        {**rag_conf, "embedding_model_name": "local-embedding", "embedding_local_path": str(missing)},
    )

    errors = validate_paths()
    assert any(e[0] == "embedding_local_path" for e in errors), "应报告路径错误"
    serialized = str(errors)
    assert str(tmp_path) not in serialized, "不得泄漏完整本地路径"
    assert "nope.bin" in serialized, "应保留文件名便于定位"

    try:
        validate_before_use("embedding")
        raise AssertionError("应抛出 ConfigValidationError")
    except ConfigValidationError as exc:
        assert str(tmp_path) not in str(exc), "validate_before_use 不得泄漏完整路径"
        assert "nope.bin" in str(exc), "应保留文件名"


def test_validate_env_vars_missing_reports_variable_name_only(monkeypatch):
    """缺失环境变量：错误只含变量名与提示，不含任何密钥形态。"""
    from utils.config_validator import validate_env_vars

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    errors = validate_env_vars()
    assert any("DEEPSEEK_API_KEY" in e[0] for e in errors)
    for _name, msg in errors:
        assert "sk-" not in msg, "校验错误不得出现密钥形态"
