# API 接口文档（API Reference）

> 后端：FastAPI（`api.main:app`），默认端口 `8000`。
> 运行后访问 **http://localhost:8000/docs** 可查看自动生成的 OpenAPI / Swagger 文档。
> 所有流式接口使用 **Server-Sent Events (SSE)**，事件体为 JSON 格式的 `AgentEvent`。

---

## 通用约定

### AgentEvent（SSE data 载荷）

```json
{
  "type": "session | message | tool_start | tool_end | plan | step | replan | report | error | done",
  "agent": "conversation | diagnostic | knowledge | orchestrator",
  "content": "文本/Markdown 内容",
  "data": { "tool": "...", "args": {}, "conversation_id": "...", "user_id": "...", "...": "..." }
}
```

> 流式响应的**首个事件固定为 `session`**，`data` 携带 `conversation_id` 与
> `user_id`，前端保存后可在后续请求中回传以延续多轮上下文。

### SSE 格式

```
event: message
data: {"type": "message", "agent": "conversation", "content": "..."}

event: message
data: {"type": "done", "agent": "conversation", "content": ""}
```

- 每个事件以 `event: message` 下发，`data` 为上述 JSON 字符串。
- 流以 `type: "done"` 的事件结束。

---

## 1. 健康检查

`GET /api/health`

返回服务状态与当前配置。

**响应示例**

```json
{
  "status": "ok",
  "model": "deepseek-v4-flash",
  "embedding": "D:\\ai_models\\...",
  "reranker_enabled": true
}
```

---

## 2. 对话（客服问答，SSE）

`POST /api/chat`

**请求体（ChatRequest）**

```json
{
  "query": "怎么清理滤网？",
  "history": [
    {"role": "user", "content": "上次的滤网"},
    {"role": "assistant", "content": "用软布擦拭即可"}
  ],
  "mode": "conversation",   // 可选，强制路由："conversation" | "diagnostic"
  "user_id": "1001",        // 可选，演示用户；未传时使用默认演示用户 1001
  "conversation_id": "conv-xxx"  // 可选，会话标识；未传时服务端新建会话
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 本轮用户提问（必填） |
| `history` | list[dict] \| null | 多轮记忆，每条含 `role` 与 `content`（未携带 `conversation_id` 时使用） |
| `mode` | string \| null | 强制路由；不填则由 Orchestrator 自动判断 |
| `user_id` | string \| null | 演示用户标识；未传时使用默认演示用户 `1001`（不再随机生成） |
| `conversation_id` | string \| null | 会话标识；未传时服务端新建并在响应中返回；归属其他用户时返回 403 |

**响应**：`text/event-stream`，首个事件为 `session`（`conversation_id` / `user_id`），
随后逐事件推送 `AgentEvent`（`tool_start` / `message` / `done` 等）。

> 携带 `conversation_id` 时，历史以**服务端会话**为准（`history` 被忽略）；
> 未携带时沿用客户端 `history`（旧客户端格式完全兼容），并自动新建会话。

---

## 3. 诊断（设备诊断，SSE）

`POST /api/diagnose`

**请求体（DiagnoseRequest）**

```json
{
  "query": "最近清洁效率很低，而且经常报告边刷被卡住",
  "user_id": "1001",           // 可选，演示用户；未传时使用默认演示用户 1001
  "conversation_id": "conv-xxx"  // 可选，会话标识；未传时服务端新建会话
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 故障描述（必填） |
| `user_id` | string \| null | 演示用户标识；未传时使用默认演示用户 `1001` |
| `conversation_id` | string \| null | 会话标识；未传时新建；归属其他用户时返回 403 |

**响应**：`text/event-stream`，首个事件为 `session`（`conversation_id` / `user_id`），依次推送：
- `plan`：排查计划（步骤列表）
- `step`：每一步执行结果与所调用的工具
- `replan`：continue / replan / end 决策
- `report`：最终 Markdown 诊断报告
- `done`：结束事件

诊断链路经 MCP Client/Server 工具层查询**模拟设备状态与运行日志**（底层为
CSV / Mock 数据源）。设备类工具失败时，报告中标记为「实时设备数据不可用」，
建议基于已有知识生成；不会向客户端泄漏内部异常、路径或密钥。

---

## 3.1 会话与身份（轻量演示级）

- **演示用户**：`user_id` 取值 `1001`-`1010`，映射到各自的模拟设备
  （见 `agent/services/device_registry.py`）；未传时使用默认演示用户 `1001`，
  不再随机生成（保证同一用户多次请求落到同一设备）。
- **会话**：`conversation_id` 由服务端生成，保存 `{user_id, messages, created_at, updated_at}`，
  仅保存对话上下文，不保存真实设备状态（设备状态仍由 MCP 工具实时查询）。
- **归属校验**：每次请求校验 `conversation_id` 是否属于当前 `user_id`；
  不归属返回 `403` + `CONVERSATION_ACCESS_DENIED`（不确认会话是否存在）。
- **存储**：当前为**单进程内存**实现，服务重启后会话失效（演示级，不引入数据库）。

---

## 4. 知识库上传

`POST /api/knowledge/upload`

**请求**：`multipart/form-data`，字段 `files`（可多个 `UploadFile`）。

**行为**：仅允许 `config/chroma.yml` 中 `allow_knowledge_file_type` 配置的扩展名（默认 `txt` / `pdf`），文件落盘到 `data/` 目录。

**响应（KnowledgeUploadResponse）**

```json
{ "success": true, "file_count": 2 }
```

---

## 5. 知识库重建

`POST /api/knowledge/rebuild`

**行为**：基于 MD5 增量重新入库 `data/` 下文档到 ChromaDB，返回当前分块总数。

**响应（KnowledgeRebuildResponse）**

```json
{ "success": true, "chunk_count": 128 }
```

> 出错时返回 HTTP 500，detail 含失败原因。

---

## 6. 人工转接（Handoff）

当用户主动要求转人工（如「转人工 / 人工客服 / 联系人工」）且 RAG 回答无法满足时，Orchestrator 拦截并自动创建人工工单。用户凭一次性 `access_key` 查询，管理员经会话 Cookie 鉴权后统一处理。

### 6.1 用户创建工单

`POST /api/handoff`

**请求体（HandoffCreateRequest）**

```json
{
  "user_question": "机器一直报错 E03，帮我转人工",
  "conversation_id": "conv-a",
  "recent_conversations": [{"role": "user", "content": "机器报错"}],
  "conversation_summary": "用户反馈故障码",
  "retrieval_sources": [{"document": "故障排除.txt", "score": 0.31}],
  "diagnostic_result": "疑似门锁传感器故障",
  "handoff_reason": "rag_low_confidence"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_question` | string | 用户问题（必填，非空） |
| `conversation_id` | string \| null | 会话关联标识，**不是访问凭证** |
| `recent_conversations` | list \| str \| null | 最近对话内容 |
| `conversation_summary` | string | 对话摘要 |
| `retrieval_sources` | list \| str \| null | RAG 来源 |
| `diagnostic_result` | string | 诊断结果 |
| `handoff_reason` | string | 转人工原因，默认 `user_request` |

**响应 `201`**：完整工单 + 一次性明文 `access_key`（仅此一次下发）。

```json
{
  "ticket_id": "c9f2...",
  "conversation_id": "conv-a",
  "user_question": "机器一直报错 E03，帮我转人工",
  "status": "pending",
  "human_reply": null,
  "handoff_reason": "rag_low_confidence",
  "created_at": "2026-08-27T10:00:00+00:00",
  "updated_at": "2026-08-27T10:00:00+00:00",
  "access_key": "k1P...43 字符"
}
```

### 6.2 用户查询工单

`GET /api/handoff/{ticket_id}?access_key=<access_key>`

- **Query**：`access_key`（用户创建时拿到的一次性凭证）。
- **响应 `200`**：工单对象（含 `human_reply`、最新 `status`），**不含 `access_key` / `access_key_hash`**。
- **响应 `404`**：凭证缺失 / 错误 / 工单不存在 → 统一 `HANDOFF_TICKET_NOT_FOUND`，不泄露工单存在性。

### 6.3 管理员登录

`POST /api/admin/login`

**请求体（AdminLoginRequest）**

```json
{ "admin_token": "<ADMIN_TOKEN>" }
```

**响应 `200`** + `Set-Cookie`：

```json
{ "success": true, "message": "登录成功", "request_id": "..." }
```

```
Set-Cookie: admin_session=<sid>; HttpOnly; Secure; SameSite=lax; Max-Age=<ttl>; Path=/
```

- Cookie 名为 **`admin_session`**，为 **HttpOnly** 管理会话 Cookie（`Secure` / `SameSite=lax` / `Max-Age` = `ADMIN_SESSION_TTL_SECONDS`，默认 8 小时）。
- 会话保存在**服务端内存**（单进程），服务重启后失效，需重新登录。
- **`401`**：令牌错误 → `HANDOFF_ADMIN_REQUIRED`。
- **`403`**：`ADMIN_TOKEN` 未配置，管理端整体禁用 → `HANDOFF_ADMIN_DISABLED`。

### 6.4 管理端工单列表

`GET /api/admin/handoffs?status=&limit=&offset=`

需要有效管理会话 Cookie（`admin_session`）。

| Query | 类型 | 说明 |
|-------|------|------|
| `status` | string \| 空 | 过滤状态：`pending` / `processing` / `resolved` / `closed` |
| `limit` | int | 页大小，默认 `20`，上限 `100` |
| `offset` | int | 偏移，默认 `0` |

**响应 `200`**（按 `created_at` 倒序）：

```json
{ "total": 3, "limit": 20, "offset": 0, "items": [ { "ticket_id": "...", "status": "pending", "...": "..." } ] }
```

### 6.5 管理端工单详情

`GET /api/admin/handoffs/{ticket_id}`

**响应 `200`**：工单详情（含 `user_question`、`conversation_summary`、`retrieval_sources`、`diagnostic_result`、`human_reply` 等完整上下文），**不回显 `access_key` / `access_key_hash`**。

**响应 `404`**：工单不存在 → `HANDOFF_TICKET_NOT_FOUND`。

### 6.6 管理端回复工单

`POST /api/admin/handoffs/{ticket_id}/reply`

**请求体（HandoffReplyRequest）**

```json
{
  "human_reply": "已为您转接售后，问题已解决。",
  "status": "resolved"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `human_reply` | string | 人工回复（必填） |
| `status` | string \| null | 更新后的状态，缺省默认 `resolved` |

**响应 `200`**：更新后的工单（`status` / `human_reply` / `updated_at` 已更新）。

### 6.7 鉴权与会话说明

- **管理会话 Cookie（`admin_session`）**：HttpOnly / Secure / SameSite=lax，前端 JS 无法读取；管理接口统一校验该 Cookie 对应的服务端会话，旧版请求头鉴权方式已弃用。
- **`access_key` 查询凭证**：仅创建时下发一次；前端仅内存保存；SQLite 只存 SHA-256 哈希（`access_key_hash`）；用户凭它查询工单状态，可在管理员回复后看到 `human_reply` 与 `resolved`。

### 6.8 错误码与状态

工单状态取值：`pending`、`processing`、`resolved`、`closed`。

| HTTP | 错误码 | 触发场景 |
|------|--------|----------|
| `401` | `HANDOFF_ADMIN_REQUIRED` | 管理会话缺失 / 无效 / 过期；`/login` 令牌错误 |
| `403` | `HANDOFF_ADMIN_DISABLED` | `ADMIN_TOKEN` 未配置，管理端整体禁用 |
| `404` | `HANDOFF_TICKET_NOT_FOUND` | 工单不存在；用户侧凭证缺失/错误统一 404 |
| `422` | — | `status` 非法、`human_reply`/`user_question` 为空等字段校验失败（FastAPI 标准 422） |
| `500` | `INTERNAL_ERROR` / `HANDOFF_TICKET_CREATE_FAILED` | 存储异常（自动建单失败在 SSE 中表现为 `error` 事件） |
| `503` | `HANDOFF_DB_NOT_CONFIGURED` | `HANDOFF_DB_PATH` 未配置，工单服务不可用 |

统一错误响应体（非 422 场景）：

```json
{ "error_code": "...", "safe_message": "...", "request_id": "..." }
```

### 6.9 安全约定

- `access_key` **不返回给管理接口**，`access_key_hash` **不对外暴露**（任何接口、SSE 事件、日志均不回显）。

---

## 7. 错误与排查

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| `/api/health` 返回异常 | 配置缺失或模型工厂初始化失败 | 检查 `config/*.yml` 与 `.env` 中的 API Key |
| SSE 流中断 | 后端异步桥接异常 | 查看后端日志（`logs/`）中的 `diagnostic_agent_error` / `sse_bridge` 条目 |
| 上传返回 `file_count: 0` | 文件扩展名不在白名单 | 检查 `chroma.yml` 的 `allow_knowledge_file_type` |
| 前端连不上后端 | `API_BASE` 指向错误 | 本地默认 `http://localhost:8000`；Docker 下应为 `http://api:8000` |

---

## 8. 调用示例（curl）

```bash
# 健康检查
curl http://localhost:8000/api/health

# 对话（SSE，携带演示用户与会话标识延续多轮上下文）
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"怎么清理滤网？","user_id":"1001","conversation_id":"conv-xxx"}'

# 诊断（SSE，首个事件为 session，返回 conversation_id）
curl -N -X POST http://localhost:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{"query":"清洁效率很低","user_id":"1001"}'

# 上传知识文档（Windows 请用绝对路径，避免 /tmp 风格路径导致读取失败）
curl -X POST http://localhost:8000/api/knowledge/upload \
  -F "files=@D:/path/to/doc.txt"

# 创建人工工单（响应含一次性 access_key）
curl -X POST http://localhost:8000/api/handoff \
  -H "Content-Type: application/json" \
  -d '{"user_question":"帮我转人工"}'

# 用户凭 access_key 查询工单
curl "http://localhost:8000/api/handoff/<ticket_id>?access_key=<access_key>"

# 管理员登录（保存 admin_session Cookie 到 cookie jar）
curl -c cookie.txt -X POST http://localhost:8000/api/admin/login \
  -H "Content-Type: application/json" \
  -d '{"admin_token":"<ADMIN_TOKEN>"}'

# 管理端列表 / 详情 / 回复（携带会话 Cookie）
curl -b cookie.txt "http://localhost:8000/api/admin/handoffs?status=pending"
curl -b cookie.txt "http://localhost:8000/api/admin/handoffs/<ticket_id>"
curl -b cookie.txt -X POST "http://localhost:8000/api/admin/handoffs/<ticket_id>/reply" \
  -H "Content-Type: application/json" \
  -d '{"human_reply":"已解决。","status":"resolved"}'
```
