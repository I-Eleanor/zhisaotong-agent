"""轻量人工工单存储（SQLite 标准库，无 ORM）。

设计要点：
- 仅使用标准库 sqlite3：与 Chroma 向量库、内存对话记忆完全隔离，互不影响；
- 路径由环境变量 HANDOFF_DB_PATH 配置，未配置时 HandoffStoreError
  （路由层转换为 503 HANDOFF_DB_NOT_CONFIGURED），绝不静默落到默认路径；
- 每次操作独立开短连接（同步 def 路由跑在线程池，连接不复用即天然线程安全），
  自动 CREATE TABLE IF NOT EXISTS，无需迁移工具；
- 用户访问凭证为创建时生成的随机 access_key（token_urlsafe(32)，高熵）：
  数据库只存 SHA-256 哈希（access_key_hash），明文仅在创建响应中出现一次；
  行转换（_row_to_ticket）始终剥离哈希列，任何接口不得回显凭证材料；
- retrieval_sources 入库序列化为 JSON 文本，读取时尽力解析回原结构，
  解析失败返回原始字符串（写入方可能是字符串也可能是列表）。
"""
import hashlib
import json
import os
import secrets
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

# 工单状态机：pending → processing → resolved / closed
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_RESOLVED = "resolved"
STATUS_CLOSED = "closed"

VALID_STATUSES: frozenset[str] = frozenset({
    STATUS_PENDING, STATUS_PROCESSING, STATUS_RESOLVED, STATUS_CLOSED,
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS handoff_tickets (
    ticket_id            TEXT PRIMARY KEY,
    access_key_hash      TEXT NOT NULL,
    conversation_id      TEXT NOT NULL,
    user_question        TEXT NOT NULL,
    recent_conversations TEXT NOT NULL DEFAULT '',
    conversation_summary TEXT NOT NULL DEFAULT '',
    retrieval_sources    TEXT NOT NULL DEFAULT '',
    diagnostic_result    TEXT NOT NULL DEFAULT '',
    handoff_reason       TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'pending',
    human_reply          TEXT,
    human_replies        TEXT,
    live_messages        TEXT NOT NULL DEFAULT '',
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_handoff_tickets_conversation
    ON handoff_tickets(conversation_id);
CREATE INDEX IF NOT EXISTS idx_handoff_tickets_status
    ON handoff_tickets(status);
"""

# access_key 随机字节数（token_urlsafe(32) → 43 字符 URL 安全文本，约 192 bit 熵）
ACCESS_KEY_BYTES = 32


class HandoffStoreError(Exception):
    """存储层内部错误（建表 / 连接失败等）。"""


class HandoffStore:
    """人工工单 SQLite 存储。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.getenv("HANDOFF_DB_PATH", "")
        if not self.db_path:
            raise HandoffStoreError("未配置 HANDOFF_DB_PATH 环境变量，无法使用工单存储")
        parent = os.path.dirname(os.path.abspath(self.db_path))
        try:
            os.makedirs(parent, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
                # 存量库补列：多轮人工回复日志 + 实时对话日志
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(handoff_tickets)")}
                if "human_replies" not in cols:
                    conn.execute("ALTER TABLE handoff_tickets ADD COLUMN human_replies TEXT")
                if "live_messages" not in cols:
                    conn.execute("ALTER TABLE handoff_tickets ADD COLUMN live_messages TEXT NOT NULL DEFAULT ''")
        except sqlite3.Error as e:
            raise HandoffStoreError(f"初始化工单数据库失败：{type(e).__name__}") from e

    # ------------------------------------------------------------------ 连接

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    # ------------------------------------------------------------------ 序列化

    @staticmethod
    def _dump_sources(retrieval_sources: Any) -> str:
        if retrieval_sources is None:
            return ""
        if isinstance(retrieval_sources, str):
            return retrieval_sources
        return json.dumps(retrieval_sources, ensure_ascii=False)

    @staticmethod
    def _loads_maybe(raw: Any) -> Any:
        if isinstance(raw, str) and raw[:1] in ("[", "{"):
            try:
                return json.loads(raw)
            except (ValueError, TypeError):
                pass
        return raw

    @staticmethod
    def _hash_access_key(access_key: str) -> str:
        """access_key 只存 SHA-256 哈希（密钥本身高熵，无需慢哈希）。"""
        return hashlib.sha256(access_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_replies(raw: Any) -> list[dict]:
        """解析多轮人工回复日志（JSON 数组）。兼容旧单条 human_reply。"""
        loaded = HandoffStore._loads_maybe(raw)
        if isinstance(loaded, list) and all(isinstance(x, dict) for x in loaded):
            return [x for x in loaded if x.get("content")]
        return []

    @staticmethod
    def _load_messages(raw: Any) -> list[dict]:
        """解析工单的实时对话日志（live_messages，JSON 数组）。"""
        loaded = HandoffStore._loads_maybe(raw)
        if isinstance(loaded, list) and all(isinstance(x, dict) for x in loaded):
            return loaded
        return []

    @staticmethod
    def _row_to_ticket(row: sqlite3.Row | dict | None) -> dict | None:
        if row is None:
            return None
        ticket = dict(row)
        # 凭证材料永不外泄：哈希列在任何对外形态中都被剥离
        ticket.pop("access_key_hash", None)
        legacy = (ticket.get("human_reply") or "").strip()
        replies = HandoffStore._load_replies(ticket.get("human_replies"))
        ticket.pop("human_replies", None)
        # 旧库仅有单条 human_reply 时补成回复历史，避免历史回复丢失
        if not replies and legacy:
            replies = [{"content": legacy, "created_at": ticket.get("updated_at")}]
        ticket["replies"] = replies
        ticket["human_reply"] = (replies[-1]["content"] if replies else None) or (legacy or None)
        ticket["live_messages"] = HandoffStore._load_messages(ticket.get("live_messages"))
        ticket["retrieval_sources"] = HandoffStore._loads_maybe(ticket.get("retrieval_sources"))
        ticket["recent_conversations"] = HandoffStore._loads_maybe(ticket.get("recent_conversations"))
        return ticket

    # ------------------------------------------------------------------ 写入

    def create(
        self,
        user_question: str,
        conversation_id: str | None = None,
        recent_conversations: Any = None,
        conversation_summary: str = "",
        retrieval_sources: Any = None,
        diagnostic_result: str = "",
        handoff_reason: str = "",
    ) -> tuple[dict, str]:
        """创建工单（status=pending），生成随机 access_key。

        返回 (工单字典[不含凭证], 明文 access_key)：明文只在创建时返回一次，
        数据库仅存哈希。conversation_id 仅用于会话关联，不是访问凭证。
        recent_conversations 为最近对话内容，序列化 JSON 存储（与
        retrieval_sources 同规则）。
        """
        access_key = secrets.token_urlsafe(ACCESS_KEY_BYTES)
        now = self._now()
        stored = {
            "ticket_id": uuid.uuid4().hex,
            "access_key_hash": self._hash_access_key(access_key),
            "conversation_id": conversation_id or uuid.uuid4().hex,
            "user_question": user_question,
            "recent_conversations": self._dump_sources(recent_conversations),
            "conversation_summary": conversation_summary or "",
            "retrieval_sources": self._dump_sources(retrieval_sources),
            "diagnostic_result": diagnostic_result or "",
            "handoff_reason": handoff_reason or "",
            "status": STATUS_PENDING,
            "human_reply": None,
            "live_messages": json.dumps([{
                "role": "user",
                "content": user_question,
                "created_at": now,
            }], ensure_ascii=False),
            "created_at": now,
            "updated_at": now,
        }
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO handoff_tickets (
                        ticket_id, access_key_hash, conversation_id, user_question,
                        recent_conversations, conversation_summary, retrieval_sources,
                        diagnostic_result, handoff_reason, status, human_reply, live_messages,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stored["ticket_id"], stored["access_key_hash"], stored["conversation_id"],
                        stored["user_question"], stored["recent_conversations"],
                        stored["conversation_summary"], stored["retrieval_sources"],
                        stored["diagnostic_result"], stored["handoff_reason"],
                        stored["status"], stored["human_reply"], stored["live_messages"],
                        stored["created_at"], stored["updated_at"],
                    ),
                )
        except sqlite3.Error as e:
            raise HandoffStoreError(f"创建工单失败：{type(e).__name__}") from e
        return self._row_to_ticket(stored), access_key

    def reply(self, ticket_id: str, human_reply: str, status: str = STATUS_RESOLVED) -> dict | None:
        """追加一条人工回复并更新状态；空回复仅更新状态（用于结束工单而不追加）。

        多轮回复会累积存储（human_replies 为 JSON 数组），不覆盖历史。
        工单不存在返回 None。
        """
        now = self._now()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT human_replies, human_reply, live_messages FROM handoff_tickets WHERE ticket_id = ?",
                    (ticket_id,),
                ).fetchone()
                if row is None:
                    return None
                replies = HandoffStore._load_replies(row["human_replies"])
                live = HandoffStore._load_messages(row["live_messages"])
                legacy = (row["human_reply"] or "").strip()
                if not replies and legacy:
                    replies = [{"content": legacy, "created_at": None}]
                text = (human_reply or "").strip()
                if text:
                    replies.append({"content": text, "created_at": now})
                    live.append({"role": "human", "content": text, "created_at": now})
                serialized = json.dumps(replies, ensure_ascii=False)
                live_serialized = json.dumps(live, ensure_ascii=False)
                last = replies[-1]["content"] if replies else None
                conn.execute(
                    "UPDATE handoff_tickets SET human_reply = ?, human_replies = ?, live_messages = ?, status = ?, updated_at = ? WHERE ticket_id = ?",
                    (last, serialized, live_serialized, status, now, ticket_id),
                )
                full = conn.execute(
                    "SELECT * FROM handoff_tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
        except sqlite3.Error as e:
            raise HandoffStoreError(f"更新工单失败：{type(e).__name__}") from e
        return self._row_to_ticket(full)

    # ------------------------------------------------------------------ 读取

    def get(self, ticket_id: str) -> dict | None:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM handoff_tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
        except sqlite3.Error as e:
            raise HandoffStoreError(f"查询工单失败：{type(e).__name__}") from e
        return self._row_to_ticket(row)

    def list_tickets(
        self,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[int, list[dict]]:
        """按 created_at 倒序分页。返回 (过滤后的总数, 当前页工单列表)。"""
        where = "WHERE status = ?" if status else ""
        params: list[Any] = [status] if status else []
        try:
            with self._connect() as conn:
                total = conn.execute(
                    f"SELECT COUNT(*) FROM handoff_tickets {where}", params  # noqa: S608
                ).fetchone()[0]
                rows = conn.execute(
                    f"""
                    SELECT * FROM handoff_tickets {where}
                    ORDER BY created_at DESC, ticket_id DESC
                    LIMIT ? OFFSET ?
                    """,  # noqa: S608
                    [*params, limit, offset],
                ).fetchall()
        except sqlite3.Error as e:
            raise HandoffStoreError(f"查询工单列表失败：{type(e).__name__}") from e
        tickets = [t for t in (self._row_to_ticket(r) for r in rows) if t is not None]
        return int(total), tickets

    def get_active_by_conversation(self, conversation_id: str) -> dict | None:
        """取某会话最近一张未结束（pending/processing）的工单；无则 None。"""
        if not conversation_id:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM handoff_tickets
                    WHERE conversation_id = ? AND status IN (?, ?)
                    ORDER BY created_at DESC, ticket_id DESC LIMIT 1
                    """,
                    (conversation_id, STATUS_PENDING, STATUS_PROCESSING),
                ).fetchone()
        except sqlite3.Error as e:
            raise HandoffStoreError(f"查询活动工单失败：{type(e).__name__}") from e
        return self._row_to_ticket(row)

    def append_user_message(self, conversation_id: str, content: str) -> dict | None:
        """人工接管期间，把用户新消息追加到活动工单的实时对话日志。

        返回更新后的工单（未找到活动工单时返回 None）。
        """
        if not conversation_id or not (content or "").strip():
            return None
        now = self._now()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM handoff_tickets
                    WHERE conversation_id = ? AND status IN (?, ?)
                    ORDER BY created_at DESC, ticket_id DESC LIMIT 1
                    """,
                    (conversation_id, STATUS_PENDING, STATUS_PROCESSING),
                ).fetchone()
                if row is None:
                    return None
                live = HandoffStore._load_messages(row["live_messages"])
                live.append({"role": "user", "content": content.strip(), "created_at": now})
                conn.execute(
                    "UPDATE handoff_tickets SET live_messages = ?, updated_at = ? WHERE ticket_id = ?",
                    (json.dumps(live, ensure_ascii=False), now, row["ticket_id"]),
                )
                full = conn.execute(
                    "SELECT * FROM handoff_tickets WHERE ticket_id = ?", (row["ticket_id"],)
                ).fetchone()
        except sqlite3.Error as e:
            raise HandoffStoreError(f"追加用户消息失败：{type(e).__name__}") from e
        return self._row_to_ticket(full)

    def get_by_access_key(self, ticket_id: str, access_key: str) -> dict | None:
        """用户侧查询凭证校验：access_key 与库中哈希匹配才返回工单。

        ticket 不存在、键为空或哈希不匹配统一返回 None（路由层一律 404，
        不泄露工单存在性）；比较使用 secrets.compare_digest 防时序侧信道。
        """
        if not ticket_id or not access_key:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM handoff_tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
        except sqlite3.Error as e:
            raise HandoffStoreError(f"查询工单失败：{type(e).__name__}") from e
        if row is None:
            return None
        stored = self._hash_access_key(access_key)
        if not secrets.compare_digest(stored, row["access_key_hash"]):
            return None
        return self._row_to_ticket(row)
