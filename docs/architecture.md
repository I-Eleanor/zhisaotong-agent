# 架构设计文档（Architecture）

> 本文档说明「智扫通」扫地机器人智能客服系统的架构设计决策、核心数据流与扩展性分析。
> 配套接口文档见 [api.md](api.md)；整体结构见根目录 [README.md](../README.md)。

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **多 Agent 协作** | 将「客服问答」「设备诊断」「知识检索」三类能力拆分为独立 Agent，各司其职、便于演进 |
| **流式体验** | 后端以 SSE 方式逐事件推送，前端实时渲染思考/工具调用/报告过程 |
| **模拟设备数据** | 诊断链路通过 MCP Client/Server 工具层访问**模拟设备状态与运行日志**（CSV / Mock 数据源），而非真实设备平台 |
| **可测试性** | 模型工厂懒加载、依赖可打桩，使整套逻辑可在无 API Key / 无 GPU 下用 pytest 验证 |
| **可部署** | 固定依赖版本，Docker Compose 一键启动 API + 前端 |

---

## 2. 总体分层

```
React 前端 ──HTTP+SSE──▶ FastAPI(API 层) ──▶ Orchestrator(路由)
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
       ConversationAgent            DiagnosticAgent                KnowledgeAgent
       (ReAct + 多轮记忆)          (Plan-Execute-Replan)          (RAG 总结包装)
                                                    │
                                                    ▼  ToolRouter（白名单路由）
                                            MCP Client（stdio 子进程适配层）
                                          DeviceMcpClient / LogMcpClient
                                                    │
                                                    ▼
                                            MCP Server(stdio)
                                          device_server / log_server
                                                    │
                                                    ▼
                                    模拟设备数据（CSV / Mock Repository）
              └─────────────────────────────┬─────────────────────────────┘
                                            ▼
                              模型层(LLM+Embedding) + 存储层(ChromaDB/会话存储)
```

---

## 3. 三 Agent 设计决策

### 3.1 Orchestrator（意图路由）

- **职责**：接收用户 query + history，决定交给哪个 Agent。
- **策略**：
  1. 关键词命中（如「故障」「报错」「不工作」「诊断」）→ 直接路由 `diagnostic`；
  2. 否则交给 LLM 做意图分类（conversation / diagnostic），避免误判。
- **决策理由**：关键词路径零延迟、可解释；LLM 兜底覆盖模糊表述。二者结合兼顾速度与召回。

### 3.2 ConversationAgent（对话 / 客服）

- **模式**：LangChain ReAct（`create_agent` + `bind_tools`）。
- **多轮记忆**：`ConversationBuffer` 分层存储，超过 `max_rounds` 后由 `Summarizer` 将早期轮次压缩为一条 system 摘要，保留最近 N 轮原文。
- **工具**：`rag_summarize`（知识库问答）等。
- **事件协议**：统一 `AgentEvent`（`type` / `content` / `data` / `agent`），通过 `_to_events` 把 LangChain 的 AIMessage / ToolMessage 翻译成前端可消费的流式事件。

### 3.3 DiagnosticAgent（诊断）

- **模式**：LangGraph `StateGraph` 编排的 **Plan-Execute-Replan** 循环。
- **四个节点**：
  | 节点 | 职责 |
  |------|------|
  | `planner_node` | 依据故障描述生成结构化排查计划（JSON 步骤列表） |
  | `executor_node` | 按步骤经 `ToolRouter` 选择诊断工具（设备状态 / 运行日志 / 错误码 / 维护 / 知识检索）并执行 |
  | `replanner_node` | 据执行结果决定 `continue` / `replan` / `end`；达上限 5 轮强制结束 |
  | `reporter_node` | 汇总结果生成 Markdown 诊断报告 |
- **迭代保护**：`MAX_ITERATIONS = 5`，防止模型陷入无限循环。
- **决策理由**：Plan-Execute-Replan 比单次 ReAct 更适合「先有方案、再逐步取证」的运维诊断场景，过程对前端透明（plan / step / replan / report 事件）。

### 3.4 KnowledgeAgent（知识库）

- **职责**：对 RAG 结果做总结包装，供 ConversationAgent 在工具调用中使用。
- **组件**：`RagSummarizeService` = ChromaDB 向量检索 + `CrossEncoderReranker` 重排 + LCEL 生成链。
- **增量入库**：`VectorStoreService.load_document()` 基于文件 MD5 去重，仅新增/变更文档入向量库。

---

## 4. 核心数据流

### 4.1 客服问答（流式）

```
用户输入(query, history)
  → Orchestrator.route → conversation
  → ConversationAgent.run
       ├─ ConversationBuffer 注入历史
       ├─ ReAct Loop: LLM → tool_call(rag_summarize) → observation → ...
       └─ 每个中间步骤 → AgentEvent
  → sse_bridge: 同步生成器 → asyncio.Queue → EventSourceResponse(SSE)
  → 前端逐事件渲染
```

### 4.2 设备诊断（Plan-Execute-Replan）

```
用户输入(query, user_id, conversation_id)
  → Orchestrator.route → diagnostic
  → DiagnosticAgent.run(query, history)
       planner → [plan 事件]（结合会话历史）
       loop(max 5):
         executor → [step 事件] → ToolRouter → MCP Client → MCP Server → CSV/Mock
         replanner → [replan 事件] → continue / replan / end
       reporter → [report 事件]
       → [done 事件]
  → SSE 推送至前端「设备诊断」Tab
```

设备状态与日志工具默认经 MCP Client 执行（`DEVICE_DATA_PROVIDER=mcp`，
默认值）；单元测试可注入 `direct` 直连 Provider 避免反复启动子进程。

### 4.3 知识库管理

```
前端上传文件 → POST /api/knowledge/upload → 落盘 data/ 目录（按扩展名白名单过滤）
前端触发重建 → POST /api/knowledge/rebuild → VectorStoreService 增量入库 → 返回 chunk 数
```

### 4.4 人工转接（Handoff）

当用户主动要求转人工（关键词命中）时，Orchestrator 拦截并自动创建人工工单，形成「用户 → 工单 → 管理员回复 → 用户回查」的完整闭环：

```
用户输入转人工关键词
  → Orchestrator 拦截 → [handoff_suggested 事件]
  → 自动创建工单（SQLite，status=pending）→ [handoff_created 事件，携带 ticket_id + access_key + status]
  → [done 事件]
  → 用户侧：前端仅内存保存 access_key
  → 管理员 POST /api/admin/login（校验 ADMIN_TOKEN）→ 下发 HttpOnly 会话 Cookie（admin_session）
  → 管理员 GET /api/admin/handoffs* 查看工单列表与上下文（对话摘要 / RAG 来源 / 诊断结果）
  → 管理员 POST /api/admin/handoffs/{ticket_id}/reply 回复 → 状态置为 resolved / processing...
  → 用户 GET /api/handoff/{ticket_id}?access_key=... 查询 → 看到 human_reply 与 resolved
```

**安全设计**

- `access_key` 仅创建时下发一次；SQLite 只存 SHA-256 哈希（`access_key_hash`）；前端仅内存保存、不渲染；管理接口与任何 SSE 事件、日志均不回显 `access_key` / `access_key_hash`。
- 管理会话 Cookie（`admin_session`）为 `HttpOnly` / `Secure` / `SameSite=lax` / `Max-Age`，服务端内存校验，旧版请求头鉴权方式已弃用。
- 管理接口统一校验服务端会话；`ADMIN_TOKEN` 未配置时管理端整体禁用（403）。

**MVP 局限**

- 管理会话为**单进程内存**存储，服务重启后失效，需重新登录；
- 跨源部署需后端 CORS `allow_credentials`，且开启凭据跨源时 `CORS_ORIGINS` **不能配置为 `*`**；
- SQLite **多 worker 写并发能力有限**；
- 不支持**实时人工聊天与 WebSocket**，用户侧通过凭 `access_key` 再次查询获取状态更新。

---

## 5. 设备数据服务边界（MCP Client/Server 与 Provider）

### 5.1 真实调用链

诊断 Agent 的设备类工具（`query_device_status` / `query_device_logs`）**默认经 MCP 执行**，
实际调用链为：

```
DiagnosticAgent
  → ToolRouter（工具白名单 + 参数补全 + 失败包装）
  → MCP Client 适配层（agent/mcp_client/：StdioServerParameters + stdio_client + ClientSession）
  → stdio MCP Server 子进程（mcp_server/device_server.py / log_server.py）
  → CsvDeviceStatusService / MockDeviceLogService（模拟数据源）
```

每个工具调用是一个完整的「启动子进程 → initialize 握手 → call_tool → 关闭」生命周期，
无长连接、无子进程泄漏。MCP Client 负责超时控制、连接失败捕获、结果文本提取，
并识别 Server 返回的「数据不可用」哨兵文本，还原为 `ServiceUnavailableError`。

> **有意设计：每次调用启动独立 stdio Server 子进程。** 这是演示级的刻意取舍：
> 每次调用新建子进程、独立事件循环、用完即关闭，换来无状态的简单性——不维护
> 长连接、不做子进程池、不处理 Server 崩溃后的重连状态机。代价是每次调用有
> 进程启动开销（毫秒级到秒级）。生产若要降本可改为长连接 / 进程复用，但这会
> 引入生命周期管理复杂度，故当前不实现。

### 5.2 Direct / MCP 双 Provider

统一接口 `DeviceDataProvider`：

```python
class DeviceDataProvider(Protocol):
    def query_status(self, user_id: str) -> str: ...
    def query_logs(self, user_id: str, days: int = 7) -> str: ...
```

- `DirectDeviceDataProvider`：进程内直连 CSV / Mock 服务（保留直接调用能力）；
- `McpDeviceDataProvider`：经 MCP Client 调用 MCP Server（主链路默认实现）。

由环境变量 `DEVICE_DATA_PROVIDER`（默认 `mcp`）或构造注入选择。二者的输出格式
（`设备ID：device-xxxx` 头 + 服务文本）与归属校验完全一致，可无缝切换。

### 5.3 为什么 MCP 作为可替换的数据服务边界

- **主链路真正走 MCP**：DiagnosticAgent 不直接 import CSV/Mock 服务，设备数据
  统一经 MCP Client 获取，服务端可独立部署、替换数据源；
- **单元测试快**：测试注入 `direct` Provider，避免反复拉起子进程；
- **失败易切换**：MCP 故障时可配置回退 `direct`，架构边界清晰；
- **耦合隔离**：MCP 调用代码集中在 `agent/mcp_client/`，不混入 ToolRouter /
  编排逻辑。

### 5.4 用户、会话、设备的区别

| 概念 | 说明 |
|------|------|
| `user_id` | 演示用户标识（默认 `1001`；注册表映射 1001-1010 到各自设备）。未传时回退默认演示用户，不再随机生成 |
| `conversation_id` | 服务端生成的会话标识，保存 `{user_id, messages, created_at, updated_at}`，仅保存上下文，不保存真实设备状态 |
| `device_id` | 经设备注册表（`agent/services/device_registry.py`）由 `user_id` 解析；`user_id → device_id → 设备状态` 链路在服务端与客户端双重校验 |

每次请求校验 `conversation_id` 是否属于当前 `user_id`，不归属则返回 403
`CONVERSATION_ACCESS_DENIED`；未知用户不发起任何子进程调用。

### 5.5 模拟数据来源

- **设备状态**：`CsvDeviceStatusService` 读取 `data/extermal/records.csv`；
- **运行日志**：`MockDeviceLogService` 基于 `user_id` 派生可复现日志。

当前项目**不包含真实设备联网、IoT 网关和云平台接入**；设备状态与运行日志均为
可复现的模拟数据。MCP Server 内部继续复用这两个服务，不另起数据源。

### 5.6 MCP 失败降级策略

MCP Client 的失败（连接失败 / 工具不存在 / 调用超时 / Server 意外退出 / 返回为空）
统一映射为 `MCPConnectionError` 族（`stage="mcp"`、`error_code=TOOL_UNAVAILABLE`），
由 ToolRouter 转换为 `StepResult(success=False)`，用户只看到固定安全文案
「实时设备数据暂时不可用，本步骤已跳过，后续建议将基于已有知识生成」。
不泄漏子进程命令、本地路径、API Key、Python 异常原文或用户内部标识。

---

## 6. SSE 桥接设计

后端 `orchestrator.execute` 是**同步生成器**，而 FastAPI 需要异步响应。为不触碰 LangChain 的异步中间件，采用桥接模式：

```
sse_bridge(gen_factory):
  queue = asyncio.Queue()
  stop  = threading.Event()
  # 线程池跑同步生成器，把每个 AgentEvent put 进 queue
  # 异步端从 queue 取事件，封装成 SSE(data=JSON) 逐条 yield
  # 生成器结束 → put None → 关闭
```

- 线程池：`ThreadPoolExecutor`
- 同步/异步边界：`asyncio.Queue` + `threading.Event`
- 优点：复用现有同步 Agent 代码，无需改造为原生 async。

---

## 7. 扩展性分析

| 扩展点 | 做法 |
|--------|------|
| **新增 Agent** | 在 `Orchestrator.route` 增加分支 + 注册新 Agent 类，前端加 Tab 即可 |
| **新增诊断工具** | 在 `agent/tools/diagnostic_tools.py` 用 `@tool` 注册，`tool_router.py` 增加 `ToolSpec` 并同步 `ALLOWED_TOOLS` 白名单 |
| **接入更多 MCP Server** | 复制 `mcp_server/*_server.py`（FastMCP + `@mcp.tool()`），在 `agent/mcp_client/` 增加对应 Client，由诊断工具以 stdio 拉起 |
| **替换设备数据源** | 实现新的 `DeviceDataProvider` 或新增 MCP Server；`DEVICE_DATA_PROVIDER` 切换，ToolRouter 无感知 |
| **替换 LLM** | 仅改 `model/factory.py` 与 `config/rag.yml`，上层无感知（依赖 `get_chat_model()` 抽象） |
| **替换向量库** | 重写 `rag/vector_store.py` 的 `VectorStoreService`，接口保持稳定 |
| **前端框架替换** | API 为纯 HTTP+SSE，任何能消费 SSE 的客户端均可对接 |

---

## 8. 关键设计取舍

- **同步 Agent + SSE 桥接** 而非全异步：降低对 LangChain 异步中间件的耦合，代码更易测试。
- **MCP Client/Server 工具层**：诊断数据来源与 Agent 逻辑解耦，主链路经 stdio 子进程访问模拟数据，可独立部署/替换数据源；配合 `direct` Provider 让单元测试不依赖子进程。
- **设备数据默认经 MCP**：`DEVICE_DATA_PROVIDER=mcp` 为生产默认，ToolRouter 不直连 CSV/Mock 服务，保证"文档说走 MCP、实际也走 MCP"。
- **懒加载模型工厂**：`get_chat_model()` / `get_embed_model()` 首次调用才实例化，导入 `api.main` 无需 API Key，便于容器启动与测试打桩。
- **Mock 友好的测试**：测试通过 monkeypatch `ChatModelFactory.generator` / `EmbeddingsFactory.generator` 注入确定性假模型，整套 pytest 零 API 费用、零网络。
