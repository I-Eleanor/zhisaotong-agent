# 智扫通 - 扫地机器人智能客服系统

> 基于 **RAG + 多 Agent（对话 / 诊断 / 知识库）** 的扫地机器人专业智能客服。后端 FastAPI + SSE 流式输出，诊断链路采用 Plan-Execute-Replan 编排，并通过 MCP Server 接入设备状态与日志数据。

---

## 项目背景

随着扫地机器人市场快速增长，用户在使用过程中面临诸多问题：
- 产品功能复杂，用户难以快速上手
- 故障排查缺乏专业指导，依赖人工客服效率低
- 使用数据分散，难以生成个性化报告

**智扫通** 应运而生 —— 一个融合 RAG 检索增强与 ReAct Agent 的智能客服系统，为用户提供 7x24 小时专业服务。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **智能问答** | 基于向量检索（ChromaDB + CrossEncoder 重排），精准回答产品使用、故障排查、选购指南等问题 |
| **设备诊断** | 诊断 Agent 采用 Plan-Execute-Replan 编排，自动生成排查计划、调用工具、复盘并产出 Markdown 诊断报告 |
| **多轮记忆** | ConversationBuffer 分层多轮记忆，超出阈值后自动摘要压缩，长对话不丢上下文 |
| **MCP 接入** | 通过 MCP Server（stdio）接入实时设备状态与运行日志，诊断数据鲜活 |
| **流式输出** | 后端 FastAPI 经 SSE 流式推送 AgentEvent，前端实时渲染 |
| **个性化报告** | 根据用户使用数据，生成专业的使用情况报告与保养建议 |

---

## 技术架构

```
┌────────────────────────────────────────────────────────────────────┐
│                          用户界面层                                  │
│       React + Vite + Tailwind 前端（客服问答 / 设备诊断 双视图）        │
└───────────────────────────────────┬────────────────────────────────┘
                                     │  HTTP + SSE  (POST /api/chat, /api/diagnose …)
                                     ▼
┌────────────────────────────────────────────────────────────────────┐
│                         API 层（FastAPI + SSE）                       │
│   /api/health  ·  /api/chat  ·  /api/diagnose  ·  /api/knowledge/*   │
│   CORS 中间件  ·  sse_bridge（同步生成器 → 异步 SSE 桥接）             │
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

> 完整设计决策与数据流见 [docs/architecture.md](docs/architecture.md)；接口字段见 [docs/api.md](docs/api.md)（或运行后访问 `/docs` 自动生成的 OpenAPI 文档）。
```

---

## 目录结构

```
Agent_project/
├── .env                    # 环境变量（API Key 等）
├── .env.example            # 环境变量示例
├── requirements.txt        # 依赖包（固定版本）
├── frontend/               # React 前端（Vite + TypeScript + Tailwind + shadcn/ui 风格）
├── Dockerfile              # API 镜像构建（Python 3.11-slim）
├── docker-compose.yml      # 一键启动 API + 前端（nginx 反代 /api）
├── archive/                # 已淘汰的 Streamlit 前端遗留快照（参考，不参与部署）
│
├── api/                    # FastAPI 后端
│   ├── main.py             # 应用入口（CORS + 路由注册 + /api/health）
│   ├── schemas.py          # 请求/响应 Pydantic 模型
│   ├── streaming.py        # SSE 桥接（同步生成器 → 异步 SSE）
│   └── routes/             # conversation / diagnostic / knowledge 路由
│
├── agent/                  # Agent 模块
│   ├── orchestrator.py     # 意图路由（关键词 + LLM 兜底）
│   ├── conversation_agent.py  # 对话 Agent（ReAct，多轮记忆）
│   ├── diagnostic_agent.py # 诊断 Agent（Plan-Execute-Replan，LangGraph）
│   ├── knowledge_agent.py  # 知识库 Agent（RAG 总结包装）
│   ├── events.py           # 统一 AgentEvent 协议
│   ├── memory/             # ConversationBuffer 分层多轮记忆 + Summarizer
│   └── tools/              # agent_tools / diagnostic_tools / middleware
│
├── mcp_server/             # MCP Server（stdio 传输）
│   ├── device_server.py    # 设备状态 / 当前用户查询
│   └── log_server.py       # 设备运行日志查询
│
├── rag/                    # RAG 模块
│   ├── rag_service.py      # RAG 检索服务
│   ├── vector_store.py     # 向量存储服务（MD5 增量入库）
│   └── reranker.py         # CrossEncoder 重排
│
├── model/                  # 模型工厂（懒加载）
│   └── factory.py          # ChatModel / Embedding 工厂
│
├── config/                 # 配置文件（rag.yml / agent.yml / chroma.yml / prompts.yml）
├── prompts/                # 提示词文件（含诊断 plan / replan / report）
├── data/                   # 知识库文档 + CSV 用户数据
├── chroma_db/              # ChromaDB 向量库（持久化）
├── logs/                   # 日志目录
│
├── tests/                  # 测试套件（pytest，覆盖率 >60%）
│   ├── conftest.py         # Mock LLM / Mock Embedding / 临时向量库
│   ├── test_rag.py         # RAG 流程测试
│   ├── test_agents.py      # Agent 行为测试
│   ├── test_tools.py       # 工具调用测试
│   ├── test_api.py         # API 接口测试
│   ├── test_mcp.py         # MCP Server 测试
│   └── test_utils.py       # 工具类测试
│
├── scripts/                # 脚本目录
├── docs/                   # 文档目录（architecture.md / api.md）
└── utils/                  # 工具模块（config / prompt_loader / path_tool / logger / file_handler）
```

---

## 核心链路说明

### 1. RAG 问答链路

```
用户提问
    │
    ▼
┌──────────────────┐
│ 1. 文档加载      │  加载 PDF/TXT 知识库
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 2. 文本分块      │  按语义切分为 chunks
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 3. Embedding     │  转换为向量表示
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 4. 存入 ChromaDB │  向量持久化存储
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 5. 相似度检索    │  根据问题检索 Top-K 相关文档
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 6. LLM 生成      │  结合上下文生成专业回答
└──────────────────┘
    │
    ▼
  返回答案
```

**代码示例：**
```python
from rag.rag_service import RagSummarizeService

rag = RagSummarizeService()
answer = rag.rag_summarize("小户型适合哪些扫地机器人？")
```

---

### 2. Agent 报告生成链路

```
用户请求："生成我的使用报告"
    │
    ▼
┌──────────────────────────┐
│ 1. 意图识别              │  Agent 判断为报告生成场景
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 2. 获取用户 ID           │  调用 get_user_id 工具
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 3. 获取当前月份          │  调用 get_current_month 工具
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 4. 注入上下文            │  调用 fill_context_for_report
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 5. 查询使用数据          │  调用 fetch_external_data
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 6. 动态切换提示词        │  Middleware 检测上下文，切换到报告提示词
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 7. LLM 生成报告          │  基于数据生成专业报告
└──────────────────────────┘
    │
    ▼
  返回 Markdown 报告
```

**关键代码：**
```python
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()  # 切换到报告提示词
    return load_system_prompts()      # 默认系统提示词
```

---

## 安装与启动

### 1. 环境要求

- Python 3.10+（本项目在 3.12 上验证通过）
- Windows / macOS / Linux

### 2. 安装依赖

建议在项目目录下创建独立的虚拟环境，避免污染系统 Python：

```bash
# 创建并激活虚拟环境（推荐放在项目内，如 .venv/ 或 venv/）
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖（固定版本）
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制示例配置
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux

# 编辑 .env 文件，填入真实 API Key
# 后端默认使用 OpenAI 兼容的 DeepSeek 接口（见 config/rag.yml）
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

> 模型与向量库路径均在 `config/rag.yml` / `config/chroma.yml` 中配置；`chat_model_name`、`deepseek_base_url`、本地 Embedding 路径按需修改。

### 4. 启动应用

后端（FastAPI + SSE）与前端（React + Vite）需分别启动：

```bash
# 终端 1：启动后端（默认 8000 端口）
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端（Vite 开发服务器，默认 5173 端口）
cd frontend
npm install        # 首次需安装依赖（已生成 package-lock.json）
npm run dev        # 启动开发服务器
```

- 前端开发访问：http://localhost:5173
- **API 文档（自动生成的 OpenAPI / Swagger）：http://localhost:8000/docs**
- 健康检查：GET http://localhost:8000/api/health
- 开发模式前端通过 Vite 的 `/api` 代理（见 `frontend/vite.config.ts`）转发到后端 8000 端口；生产构建（`npm run build`）由 `frontend/nginx.conf` 经 nginx 反代 `/api`

> 也可使用 Docker 一键启动，见文末「Docker 部署」。

---

## 典型对话示例

### 示例 1：产品咨询

```
用户：小户型适合哪些扫地机器人？

Agent 思考：用户询问小户型选购建议，需要从知识库检索专业内容。

[调用工具] rag_summarize(query="小户型 扫地机器人 选购")

Agent 回答：
针对小户型，建议选择以下类型的扫地机器人：

1. **紧凑型设计**：选择机身小巧、转弯灵活的型号，如石头 P10、追觅 X40
2. **避障能力强**：小户型家具密集，需要优秀的避障能力避免卡困
3. **噪音控制好**：小户型空间有限，低噪音（<65dB）更舒适
4. **续航适中**：小户型清洁面积小，无需超长续航，1500mAh 即可

推荐型号：石头 P10 Pro、追觅 X40 Pro、科沃斯 T30...
```

---

### 示例 2：报告生成

```
用户：生成我的使用报告

Agent 思考：用户需要生成个人使用报告，按照固定流程执行。

[调用工具] get_user_id()
[调用工具] get_current_month()
[调用工具] fill_context_for_report()
[调用工具] fetch_external_data(user_id="1001", month="2025-06")

Agent 回答：
# 黑马程序员扫地机器人使用情况报告与保养建议

## 使用概况
- **用户 ID**：1001
- **报告月份**：2025年6月
- **清洁效率**：90%
- **耗材状态**：中等消耗

## 使用分析
本月您的扫地机器人表现优秀，清洁效率较上月提升5%...
建议：定期清理滚刷，保持最佳性能...

## 保养建议
1. 每周清理尘盒一次
2. 每月检查边刷磨损情况
3. 建议更换滤芯周期：3个月
```

---

## 配置说明

### RAG 配置 (config/rag.yml)

```yaml
chat_model_name: deepseek-v4-flash          # OpenAI 兼容的对话模型
deepseek_base_url: https://api.deepseek.com/v1   # 通过环境变量 DEEPSEEK_API_KEY 注入密钥
embedding_local_path: "D:\\ai_models\\..." # 本地 Sentence-Transformers 路径（或留空走默认）
reranker_model: <cross-encoder 模型名>     # CrossEncoder 重排（可选）
reranker_enabled: true
```

### Agent 配置 (config/agent.yml)

```yaml
external_data_path: data/extermal/records.csv
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **前端** | React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui 风格组件（SSE 实时渲染） |
| **后端** | FastAPI + SSE（sse-starlette） |
| **Agent 框架** | LangChain（ReAct）+ LangGraph（Plan-Execute-Replan） |
| **编排/路由** | Orchestrator（关键词 + LLM 兜底意图分类） |
| **协议** | MCP（Model Context Protocol，stdio 接入设备状态/日志） |
| **LLM** | DeepSeek（OpenAI 兼容接口，可切换） |
| **Embedding** | Sentence-Transformers（本地） |
| **重排** | CrossEncoder（CrossEncoderReranker） |
| **向量数据库** | ChromaDB |
| **配置管理** | PyYAML + python-dotenv |
| **测试** | pytest + pytest-asyncio + pytest-cov |

---

## 项目亮点

1. **三 Agent 协作架构**：对话 Agent（ReAct）+ 诊断 Agent（Plan-Execute-Replan）+ 知识库 Agent（RAG 总结），职责清晰、可独立扩展
2. **Orchestrator 意图路由**：关键词命中优先、LLM 兜底分类，兼顾响应速度与准确率
3. **分级多轮记忆**：ConversationBuffer 分层存储，超阈值自动摘要压缩，长对话不丢上下文
4. **MCP 实时数据接入**：通过 stdio 协议的 MCP Server 接入设备状态与运行日志，诊断依据鲜活
5. **SSE 流式输出**：后端同步生成器经 `sse_bridge` 桥接为异步 SSE，前端实时渲染 AgentEvent
6. **完整测试覆盖**：pytest 套件（Mock LLM / Mock Embedding）覆盖 RAG、Agent、工具、API、MCP，覆盖率 > 60%
7. **固定依赖版本 + Docker 化**：requirements 锁版本，docker-compose 一键启动 API + 前端
8. **规范目录结构**：api/、agent/、mcp_server/、rag/、tests/、docs/ 分层，易于维护

---

## 测试

项目提供零外部依赖（不调用真实 LLM / Embedding）的确定性测试套件：

```bash
# 安装测试依赖（已含于 requirements.txt）
pip install -r requirements.txt

# 运行全部测试
pytest tests/ -v

# 生成覆盖率报告（要求 > 60%）。推荐用 coverage run + report：
# ⚠️ 说明：pytest --cov 会走 parallel 模式的 combine/erase，在受限沙箱里会被安全删除策略拦截；
#        在普通开发机上 pytest --cov 可正常使用。此处统一用 coverage run 以兼容沙箱。
coverage run -m pytest tests/ -q
coverage report                 # 输出各模块覆盖率与总计（>60% 达标）
# 等价一行：coverage run -m pytest tests/ -q && coverage report --show-missing

# 运行特定测试文件
pytest tests/test_agents.py -v
```

测试策略：
- **Mock LLM**：`ScriptedChatModel` 按提示词返回脚本化回复，驱动诊断 Agent 走完 plan→execute→replan→report
- **Mock Embedding**：`FakeEmbeddings` 用 sha256 生成确定性向量，免去模型权重
- **临时向量库**：每个用例创建临时 ChromaDB 目录，测试后自动清理
- **API / MCP**：API 在路由层用桩 Orchestrator 替换；MCP 通过 `StdioServerParameters` 真实拉起 Server 验证工具列表与调用

---

## 后续规划

- [x] 多 Agent 架构（对话 / 诊断 / 知识库）
- [x] 多轮对话记忆
- [x] 诊断 Agent（Plan-Execute-Replan）
- [x] MCP 接入设备状态 / 日志
- [x] API 接口（FastAPI + SSE）
- [x] 测试套件 + 文档 + Docker 部署
- [ ] 接入更多 LLM 后端（OpenAI、通义等）
- [ ] 用户反馈 / 评价机制
- [ ] 前端鉴权与多用户会话隔离

---

## Docker 部署

使用 docker-compose 可一键启动 **API（8000）+ 前端（8511，React + nginx）** 两个服务：

```bash
# 构建并后台启动
docker compose up -d --build

# 查看日志
docker compose logs -f
```

- 前端访问：http://localhost:8511
- API 文档：http://localhost:8000/docs
- 前端容器由 nginx 提供静态文件，并将 `/api` 反代到 `api` 服务（8000），见 `frontend/nginx.conf`
- 密钥通过 `.env` 注入 API 服务（`env_file: .env`）

> 前端镜像基于 `frontend/Dockerfile`（多阶段：node 构建 → nginx 运行），后端镜像基于根 `Dockerfile`（Python 3.11-slim）。

### 无 Docker 引擎时的本地验证

本机若没有运行中的 Docker 引擎（`docker info` 不可用），无法 `docker compose up`，但可先用以下方式校验配置与启动命令是否正确：

```bash
# 1) 仅校验 compose 文件语法（无需 daemon，客户端即可完成）
docker compose config -q && echo "compose 配置合法"

# 2) 按容器内相同的命令在本地实跑一遍（启动 API + 前端并探活）
python scripts/deploy_smoke.py        # 退出码 0 = 通过
```

`scripts/deploy_smoke.py` 会按 `docker-compose.yml` 中的命令拉起
`uvicorn api.main:app`，并用 `vite preview` 提供 `frontend/dist` 静态前端，
探测 `/api/health` 与前端端口，等价于把"容器要干的事"在本地验证一遍。

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。