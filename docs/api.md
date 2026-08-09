# API 接口文档（API Reference）

> 后端：FastAPI（`api.main:app`），默认端口 `8000`。
> 运行后访问 **http://localhost:8000/docs** 可查看自动生成的 OpenAPI / Swagger 文档。
> 所有流式接口使用 **Server-Sent Events (SSE)**，事件体为 JSON 格式的 `AgentEvent`。

---

## 通用约定

### AgentEvent（SSE data 载荷）

```json
{
  "type": "message | tool_start | tool_end | plan | step | replan | report | error | done",
  "agent": "conversation | diagnostic | knowledge",
  "content": "文本/Markdown 内容",
  "data": { "tool": "...", "args": {}, "...": "..." }
}
```

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
  "mode": "conversation"   // 可选，强制路由："conversation" | "diagnostic"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 本轮用户提问（必填） |
| `history` | list[dict] \| null | 多轮记忆，每条含 `role` 与 `content` |
| `mode` | string \| null | 强制路由；不填则由 Orchestrator 自动判断 |

**响应**：`text/event-stream`，逐事件推送 `AgentEvent`（`tool_start` / `message` / `done` 等）。

---

## 3. 诊断（设备诊断，SSE）

`POST /api/diagnose`

**请求体（DiagnoseRequest）**

```json
{
  "query": "最近清洁效率很低，而且经常报告边刷被卡住"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 故障描述（必填） |

**响应**：`text/event-stream`，依次推送：
- `plan`：排查计划（步骤列表）
- `step`：每一步执行结果与所调用的工具
- `replan`：continue / replan / end 决策
- `report`：最终 Markdown 诊断报告
- `done`：结束事件

诊断链路会经 MCP Server 查询真实设备状态与运行日志。

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

## 6. 错误与排查

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| `/api/health` 返回异常 | 配置缺失或模型工厂初始化失败 | 检查 `config/*.yml` 与 `.env` 中的 API Key |
| SSE 流中断 | 后端异步桥接异常 | 查看后端日志（`logs/`）中的 `diagnostic_agent_error` / `sse_bridge` 条目 |
| 上传返回 `file_count: 0` | 文件扩展名不在白名单 | 检查 `chroma.yml` 的 `allow_knowledge_file_type` |
| 前端连不上后端 | `API_BASE` 指向错误 | 本地默认 `http://localhost:8000`；Docker 下应为 `http://api:8000` |

---

## 7. 调用示例（curl）

```bash
# 健康检查
curl http://localhost:8000/api/health

# 对话（SSE）
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"怎么清理滤网？"}'

# 诊断（SSE）
curl -N -X POST http://localhost:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{"query":"清洁效率很低"}'

# 上传知识文档（Windows 请用绝对路径，避免 /tmp 风格路径导致读取失败）
curl -X POST http://localhost:8000/api/knowledge/upload \
  -F "files=@D:/path/to/doc.txt"
```
