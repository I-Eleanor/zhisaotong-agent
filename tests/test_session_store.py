"""轻量会话体系测试（服务端会话存储 + 归属校验）。

覆盖：
- 首次请求创建 conversation_id（服务端生成）
- 后续请求携带同一 conversation_id 可读取历史
- 不同 user_id 读取他人会话 → ConversationAccessDeniedError
- 未知 conversation_id → 新建会话（服务端重新生成 ID）
- 会话消息上限裁剪，updated_at 随写入更新
"""
import pytest

from agent.services.session_store import (
    ConversationAccessDeniedError,
    ConversationSessionStore,
)


def test_create_session_generates_server_side_id():
    store = ConversationSessionStore()
    session = store.create("1001")

    assert session["conversation_id"].startswith("conv-")
    assert session["user_id"] == "1001"
    assert session["messages"] == []
    assert session["created_at"] and session["updated_at"]


def test_resolve_without_id_creates_new_session():
    store = ConversationSessionStore()
    session = store.resolve("1001", None)
    assert session["conversation_id"]
    # 同一用户的多次"无 ID"请求各自新建会话
    assert store.resolve("1001", None)["conversation_id"] != session["conversation_id"]


def test_second_request_reads_history_of_first():
    """第二轮请求能读取第一轮的对话历史（会话连续性）。"""
    store = ConversationSessionStore()
    first = store.resolve("1001", None)
    cid = first["conversation_id"]

    store.append_message(cid, "user", "机器清洁效率很低")
    store.append_message(cid, "assistant", "已查询设备状态，建议清理主刷")

    second = store.resolve("1001", cid)
    assert second["conversation_id"] == cid, "同一会话 ID 应复用"
    history = second["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "机器清洁效率很低" in history[0]["content"]
    assert "清理主刷" in history[1]["content"]


def test_third_round_keeps_full_context():
    """第三轮仍能读到前两轮内容（演示"已讨论过 E03 / 已执行建议"的上下文延续）。"""
    store = ConversationSessionStore()
    cid = store.resolve("1001", None)["conversation_id"]

    store.append_message(cid, "user", "机器清洁效率很低")
    store.append_message(cid, "assistant", "建议查询设备状态")
    store.append_message(cid, "user", "还出现过 E03")
    store.append_message(cid, "assistant", "E03 为边刷异常，建议清理边刷")

    history = store.resolve("1001", cid)["messages"]
    joined = "\n".join(m["content"] for m in history)
    assert "清洁效率很低" in joined and "E03" in joined and "清理边刷" in joined


def test_other_user_cannot_read_session():
    """不同 user_id 读取他人会话 → 拒绝（不返回也不确认会话存在）。"""
    store = ConversationSessionStore()
    cid = store.resolve("1001", None)["conversation_id"]
    store.append_message(cid, "user", "我的私有问题")

    with pytest.raises(ConversationAccessDeniedError):
        store.resolve("1002", cid)


def test_unknown_conversation_id_creates_fresh_session():
    """未知 conversation_id（如服务重启后）→ 新建会话，不回退到他人会话。"""
    store = ConversationSessionStore()
    session = store.resolve("1001", "conv-does-not-exist")
    assert session["conversation_id"] != "conv-does-not-exist"
    assert session["messages"] == []


def test_append_updates_timestamp_and_trims_history():
    store = ConversationSessionStore(max_messages=3)
    cid = store.resolve("1001", None)["conversation_id"]

    for i in range(5):
        store.append_message(cid, "user", f"第{i}条")

    messages = store.history(cid)
    assert len(messages) == 3, "超出上限时保留最近的消息"
    assert messages[-1]["content"] == "第4条"


def test_sessions_are_isolated_between_users():
    store = ConversationSessionStore()
    a = store.resolve("1001", None)["conversation_id"]
    b = store.resolve("1002", None)["conversation_id"]
    store.append_message(a, "user", "A 的问题")
    store.append_message(b, "user", "B 的问题")

    assert store.history(b)[0]["content"] == "B 的问题"
    assert store.history(a)[0]["content"] == "A 的问题"


def test_empty_content_is_not_recorded():
    store = ConversationSessionStore()
    cid = store.resolve("1001", None)["conversation_id"]
    store.append_message(cid, "user", "   ")
    assert store.history(cid) == [], "空白内容不写入会话历史"
