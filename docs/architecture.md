# 架构设计文档（Architecture）

> 本文档说明「智扫通」扫地机器人智能客服系统的架构设计决策、核心数据流与扩展性分析。
> 配套接口文档见 [api.md](api.md)；整体结构见根目录 [README.md](../README.md)。

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **多 Agent 协作** | 将「客服问答」「设备诊断」「知识检索」三类能力拆分为独立 Agent，各司其职、便于演进 |
| **流式体验** | 后端以 SSE 方式逐事件推送，前端实时渲染思考/工具调用/报告过程 |
| **实时数据** | 诊断链路通过 MCP 接入真实设备状态与日志，而非仅依赖静态知识库 |
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
                                                    ▼
                                            MCP Server(stdio)
                                          device / log 工具
              └─────────────────────────────┬─────────────────────────────┘
                                            ▼
                              模型层(LLM+Embedding) + 存储层(ChromaDB/CSV/文档)
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
  | `executor_node` | 按步骤语义选择诊断工具（设备状态 / 错误码 / 维护 / 知识检索）并执行 |
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
用户输入(query)
  → Orchestrator.route → diagnostic
  → DiagnosticAgent.run
       planner → [plan 事件]
       loop(max 5):
         executor → [step 事件] → 调用 MCP/诊断工具
         replanner → [replan 事件] → continue / replan / end
       reporter → [report 事件]
       → [done 事件]
  → SSE 推送至前端「设备诊断」Tab
```

### 4.3 知识库管理

```
前端上传文件 → POST /api/knowledge/upload → 落盘 data/ 目录（按扩展名白名单过滤）
前端触发重建 → POST /api/knowledge/rebuild → VectorStoreService 增量入库 → 返回 chunk 数
```

---

## 5. SSE 桥接设计

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

## 6. 扩展性分析

| 扩展点 | 做法 |
|--------|------|
| **新增 Agent** | 在 `Orchestrator.route` 增加分支 + 注册新 Agent 类，前端加 Tab 即可 |
| **新增诊断工具** | 在 `agent/tools/diagnostic_tools.py` 用 `@tool` 注册，`executor_node._select_tool` 增加语义映射 |
| **接入更多 MCP Server** | 复制 `mcp_server/*_server.py`（FastMCP + `@mcp.tool()`），由诊断工具以 stdio 拉起 |
| **替换 LLM** | 仅改 `model/factory.py` 与 `config/rag.yml`，上层无感知（依赖 `get_chat_model()` 抽象） |
| **替换向量库** | 重写 `rag/vector_store.py` 的 `VectorStoreService`，接口保持稳定 |
| **前端框架替换** | API 为纯 HTTP+SSE，任何能消费 SSE 的客户端均可对接 |

---

## 7. 关键设计取舍

- **同步 Agent + SSE 桥接** 而非全异步：降低对 LangChain 异步中间件的耦合，代码更易测试。
- **MCP stdio 而非内嵌函数**：诊断数据来源与 Agent 逻辑解耦，可独立部署/替换数据源。
- **懒加载模型工厂**：`get_chat_model()` / `get_embed_model()` 首次调用才实例化，导入 `api.main` 无需 API Key，便于容器启动与测试打桩。
- **Mock 友好的测试**：测试通过 monkeypatch `ChatModelFactory.generator` / `EmbeddingsFactory.generator` 注入确定性假模型，整套 pytest 零 API 费用、零网络。
