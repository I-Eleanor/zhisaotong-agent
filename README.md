# 智扫通 - 扫地机器人智能客服系统

> RAG + 多 Agent 协作的扫地机器人专业智能客服。FastAPI + SSE 流式输出，诊断链路采用 Plan-Execute-Replan 编排，MCP Server 接入实时设备数据。

**89 测试用例 · 覆盖率 ≥70% · CI 自动化 · Docker 一键部署**

---

## 解决的问题

| 痛点 | 智扫通方案 |
|------|-----------|
| 产品功能复杂，用户难以快速上手 | RAG 检索增强（ChromaDB + CrossEncoder 重排），精准回答使用/选购/保养问题 |
| 故障排查缺乏专业指导，依赖人工客服 | 诊断 Agent 自动生成排查计划 → 调用工具 → 复盘 → 输出 Markdown 诊断报告 |
| 使用数据分散，难以生成个性化报告 | Agent 自动查询使用数据，生成专业报告与保养建议 |
| 长对话上下文丢失 | ConversationBuffer 分层记忆，超阈值自动摘要压缩 |

---

## 核心功能

- **智能问答** — 向量检索 + CrossEncoder 重排，回答产品使用、故障排查、选购指南
- **设备诊断** — Plan-Execute-Replan 编排，自动生成排查计划、调用工具、产出诊断报告
- **个性化报告** — 根据用户使用数据，生成使用情况报告与保养建议
- **多轮记忆** — 分层 ConversationBuffer，超出阈值自动摘要压缩
- **MCP 实时接入** — 通过 MCP Server（stdio）接入设备状态与运行日志
- **流式输出** — FastAPI + SSE 流式推送 AgentEvent，前端实时渲染

---

## 系统架构

```
┌────────────────────────────────────────────────────────────────────┐
│                          用户界面层                                  │
│       React + Vite + Tailwind 前端（客服问答 / 设备诊断 双视图）        │
└───────────────────────────────────┬────────────────────────────────┘
                                     │  HTTP + SSE
                                     ▼
┌────────────────────────────────────────────────────────────────────┐
│                         API 层（FastAPI + SSE）                       │
│   /api/health  ·  /api/chat  ·  /api/diagnose  ·  /api/knowledge/*   │
│   限流 · Token 鉴权 · 请求大小限制 · request_id 追踪 · CORS            │
└───────────────────────────────────┬────────────────────────────────┘
                                     │  get_orchestrator()
                                     ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Orchestrator（意图路由）                        │
│   关键词（故障/报错…）→ diagnostic  ｜  否则 → LLM 分类 → conversation │
└───────┬───────────────────────────────────────────┬─────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────────────┐         ┌──────────────────────────────────────┐
│   ConversationAgent        │         │   DiagnosticAgent（Plan-Execute-Replan）│
│   （ReAct，多轮记忆）       │         │   planner → executor → replanner → reporter │
│   └─ KnowledgeAgent        │         └───────────────┬──────────────────────┘
│      （RAG 总结）          │                          │ 工具调用
└───────────┬───────────────┘                          │
            │                                          ▼
            │                          ┌──────────────────────────────────────┐
            │                          │  MCP Server（stdio）                  │
            │                          │  • device_server：设备状态/耗材/效率   │
            │                          │  • log_server：运行日志               │
            │                          └──────────────────────────────────────┘
            ▼
┌────────────────────────────────────────────────────────────────────┐
│                           模型 / 存储层                                │
│  LLM：DeepSeek（OpenAI 兼容） ｜ Embedding：本地 Sentence-Transformers │
│  ChromaDB（向量知识库 + CrossEncoder 重排） ｜ 设备数据 CSV ｜ 知识库文档  │
└────────────────────────────────────────────────────────────────────┘
```

> 完整设计决策与数据流见 [docs/architecture.md](docs/architecture.md)；接口字段见 [docs/api.md](docs/api.md)（或运行后访问 `/docs`）。

---

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Agent 编排 | LangGraph StateGraph | Plan-Execute-Replan 循环需要状态机，LangGraph 原生支持 |
| SSE 桥接 | ThreadPoolExecutor + asyncio.Queue | LangChain Agent 为同步生成器，桥接为异步 SSE 无需改造核心代码 |
| 意图路由 | 关键词优先 + LLM 兜底 | 关键词命中零延迟，LLM 兜底保证召回，兼顾速度与准确率 |
| Embedding | 本地 Sentence-Transformers | 避免网络依赖和 API 费用，离线可用 |
| 重排 | CrossEncoder Reranker | 向量召回后精排，提升 Top-K 相关性 |
| 增量入库 | MD5 去重 | 避免重复入库，支持知识库热更新 |
| 模型加载 | 懒加载工厂 | 导入 `api.main` 无需 API Key，容器启动与测试打桩解耦 |
| 测试策略 | Mock LLM + FakeEmbeddings | 零 API 费用、零网络、确定性结果 |

---

## 快速启动

### 环境要求

- Python 3.12+ · Node 20+ · Windows / macOS / Linux

### 本地开发

```bash
# 1. 创建虚拟环境
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt          # 运行时依赖
pip install -r requirements-dev.txt      # 开发依赖（测试/代码质量）

# 3. 配置环境变量
copy .env.example .env                   # Windows
# cp .env.example .env                   # macOS/Linux
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 4. 启动后端
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. 启动前端（新终端）
cd frontend && npm install && npm run dev
```

- 前端：http://localhost:5173
- API 文档：http://localhost:8000/docs
- 健康检查：`GET /api/health`

### Docker 部署

```bash
docker compose up -d --build
```

- 前端：http://localhost:8511 · API：http://localhost:8000/docs

---

## 配置说明

### 环境变量（.env）

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek 对话模型密钥（OpenAI 兼容接口） |
| `DASHSCOPE_API_KEY` | ❌ | 在线 Embedding 备选（当前走本地，可留空） |
| `CORS_ORIGINS` | ❌ | 允许的跨域来源，默认 `*` |
| `API_TOKEN` | ❌ | API 鉴权 Token，留空不强制鉴权 |
| `HANDOFF_DB_PATH` | ❌ | 人工工单 SQLite 路径，未配置时工单接口返回 503 |
| `ADMIN_TOKEN` | ❌ | 管理端登录令牌（`POST /api/admin/login` 校验），未配置时管理接口与登录整体禁用 403 |
| `ADMIN_SESSION_TTL_SECONDS` | ❌ | 管理会话有效期（秒），默认 8 小时 |
| `LLM_TIMEOUT_SECONDS` | ❌ | LLM 调用超时，默认 60 |

### 核心配置文件

| 文件 | 内容 |
|------|------|
| `config/rag.yml` | 对话模型、Embedding 路径、Reranker 开关 |
| `config/chroma.yml` | 向量库路径、chunk_size/overlap、Top-K、重排参数 |
| `config/agent.yml` | 外部数据路径、多轮记忆轮数与摘要压缩 |
| `config/prompts.yml` | 提示词文件路径映射 |

---

## 测试与 CI

### 测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                        # 89 用例，零外部依赖
coverage run -m pytest tests/ -q && coverage report   # 覆盖率 ≥70%
```

测试策略：Mock LLM（`ScriptedChatModel`）+ FakeEmbeddings（`sha256` 确定性向量）+ 临时 ChromaDB，零 API 费用、零网络。

### CI（GitHub Actions）

```
push / PR → backend（ruff → mypy → pytest → coverage ≥70%）
          → frontend（eslint → tsc → vite build）
          → docker（compose config → compose build）
```

---

## RAG 评测

评测框架位于 `eval/`，包含 40 条多类别评测样本（事实查询/操作步骤/故障诊断/多文档综合/模糊表达/无答案/干扰文档/多轮追问）。

### 检索能力评测

```bash
python eval/eval_retrieval.py --split test
```

指标：Hit Rate@K · Recall@K · MRR · 平均/P95 延迟。自动运行「向量召回 + Reranker」与「仅向量召回」对照实验。

### 答案质量评测

```bash
python eval/eval_answer.py --split test
```

维度：关键词命中率 · 来源命中率 · 拒答准确率 · LLM-as-Judge（正确性/忠实度/完整性/引用准确率，1-5 分）。

> 运行评测后结果输出到 `eval/results/`。对照实验数据可直接用于简历量化描述。

---

## 可靠性设计

| 机制 | 实现 |
|------|------|
| 统一异常体系 | `utils/exceptions.py` — ConfigurationError / ModelInvocationError / RetrievalError / StreamExecutionError 等，区分可重试与不可重试 |
| 超时与重试 | `utils/resilience.py` — 指数退避重试，仅对网络超时/429/部分 5xx 重试 |
| 请求级追踪 | `utils/request_context.py` — request_id 贯穿 FastAPI → Orchestrator → Agent → RAG → 日志 |
| 结构化日志 | JSON 格式，含 request_id / event / duration_ms / 模块标签 |
| 健康检查分级 | `/api/health/live`（存活探针）· `/api/health/ready`（就绪探针，检查向量库+模型） |
| API 安全 | 限流（slowapi）· Token 鉴权 · 请求体大小限制 · 文件上传白名单/大小/数量限制 |
| 配置安全 | CORS 来源从环境变量读取 · .env.example 仅占位符 · 日志脱敏 API Key |

---

## 人工兜底机制

当 RAG 无法回答用户问题且用户主动要求人工时，系统将对话转接为「人工工单」，供管理员统一查看、跟进与回复。当前为**轻量 MVP**：单管理员、文本回复与状态流转，不包含实时人工聊天与 WebSocket。

### 触发与自动建单

1. **用户主动输入转人工关键词**（如「转人工 / 人工客服 / 联系人工」）被 Orchestrator 拦截；
2. 系统**自动创建人工工单**并持久化到 SQLite（`HANDOFF_DB_PATH`）；
3. 事件流依次下发 `handoff_suggested`、`handoff_created`（后随 `done`）；
4. `handoff_created` 事件携带 `ticket_id`、`status`（`pending`）以及一次性 `access_key`。

### access_key 凭证保护

- `access_key` **仅通过创建流程下发一次**（`POST /api/handoff` 的 201 响应，或 SSE `handoff_created` 事件）；
- 前端拿到后在**内存**保存，仅用于后续凭它查询工单，**不渲染、不落盘**；
- SQLite 中**只保存 `access_key` 的 SHA-256 哈希**（`access_key_hash`），明文不落库、不写日志、不进入任何工单响应。

### 管理员会话与回复

1. 管理员调用 **`POST /api/admin/login`**（提交 `admin_token` = 环境变量 `ADMIN_TOKEN`）登录；
2. 登录成功后端下发 **HttpOnly 管理会话 Cookie**（`admin_session`，含 `Secure` / `SameSite` / `Max-Age`），前端 JS 无法读取；
3. 经管理接口**查看工单列表、查看上下文、输入人工回复**，回复后状态可置为 `resolved` / `processing` 等；
4. 用户通过 **`GET /api/handoff/{ticket_id}?access_key=...`** 查询工单状态，可在管理员回复后看到 `human_reply` 与最新状态。

接口细节见 [docs/api.md](docs/api.md#6-人工转接handoff)，完整流程见 [docs/architecture.md](docs/architecture.md#44-人工转接handoff)。

### 注意事项（MVP 局限）

- 管理会话为**单进程内存**存储，服务重启后失效，需重新登录；
- 跨源（不同端口/域名）部署时后端需开启 CORS `allow_credentials`；
- 开启带凭据跨源时 `CORS_ORIGINS` **不能配置为 `*`**，必须为显式来源列表；
- SQLite **多 worker 写并发能力有限**，高并发工单写入需更换数据库或加锁策略；
- 当前为人工工单的**非实时**模型，不支持实时人工聊天与 WebSocket 推送。

---

## 项目结构

```
Agent_project/
├── api/                    # FastAPI 后端（main / schemas / streaming / routes / security）
├── agent/                  # Agent 模块
│   ├── orchestrator.py     # 意图路由（关键词 + LLM 兜底）
│   ├── conversation_agent.py  # 对话 Agent（ReAct + 多轮记忆）
│   ├── diagnostic_agent.py # 诊断 Agent（Plan-Execute-Replan，LangGraph）
│   ├── knowledge_agent.py  # 知识库 Agent（RAG 总结包装）
│   ├── events.py           # 统一 AgentEvent 协议
│   ├── memory/             # ConversationBuffer + Summarizer
│   └── tools/              # agent_tools / diagnostic_tools / middleware
├── mcp_server/             # MCP Server（stdio）：device_server / log_server
├── rag/                    # RAG 模块（rag_service / vector_store / reranker）
├── model/                  # 模型工厂（懒加载）
├── config/                 # 配置文件（rag / agent / chroma / prompts）
├── prompts/                # 提示词文件
├── data/                   # 知识库文档 + CSV 用户数据
├── eval/                   # RAG 评测（dataset / eval_retrieval / eval_answer）
├── tests/                  # 测试套件（89 用例，覆盖率 ≥70%）
├── utils/                  # 工具模块（config / exceptions / resilience / request_context / logger / file_handler）
├── frontend/               # React 前端（Vite + TypeScript + Tailwind + shadcn/ui 风格）
├── scripts/                # 脚本（dev / deploy_smoke / rebuild_vectorstore 等）
├── docs/                   # 文档（architecture.md / api.md）
├── Dockerfile              # 多阶段构建（builder + runtime，非 root 用户）
├── docker-compose.yml      # 一键部署（volume 持久化 + 资源限制 + 健康检查）
├── requirements.txt        # 生产依赖
└── requirements-dev.txt    # 开发依赖（测试/代码质量）
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 前端 | React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui 风格 |
| 后端 | FastAPI + SSE（sse-starlette） |
| Agent | LangChain（ReAct）+ LangGraph（Plan-Execute-Replan） |
| 编排 | Orchestrator（关键词 + LLM 兜底意图分类） |
| 协议 | MCP（Model Context Protocol，stdio） |
| LLM | DeepSeek（OpenAI 兼容接口，可切换） |
| Embedding | Sentence-Transformers（本地） |
| 重排 | CrossEncoder（CrossEncoderReranker） |
| 向量库 | ChromaDB |
| 测试 | pytest + pytest-asyncio + pytest-cov（89 用例，≥70%） |
| 代码质量 | ruff + mypy |
| CI | GitHub Actions（backend + frontend + docker） |

---

## 已知限制

- Embedding 和 Reranker 模型路径硬编码在 `config/rag.yml` / `config/chroma.yml`，跨机器部署需手动修改
- MCP Server 仅支持 stdio 传输，不支持远程 MCP
- 多用户会话隔离尚未实现，当前为单用户演示
- 评测结果尚未运行，`eval/results/` 为空

---

## 后续规划

- [ ] 接入更多 LLM 后端（OpenAI、通义等）
- [ ] 用户反馈 / 评价机制
- [ ] 前端鉴权与多用户会话隔离
- [ ] Embedding 模型路径配置化（支持环境变量或自动下载）
- [ ] 运行 RAG 评测并记录基线数据
- [ ] Prometheus 指标导出（`/metrics`）

---

## 许可证

MIT License
