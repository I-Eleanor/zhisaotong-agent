# 智扫通 - 扫地机器人智能客服系统

> 基于 RAG + ReAct Agent 的扫地机器人专业智能客服，支持知识问答与个性化报告生成

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
| **智能问答** | 基于向量检索，精准回答产品使用、故障排查、选购指南等问题 |
| **环境适配** | 自动获取用户位置与天气，判断是否适合使用扫地机器人 |
| **个性化报告** | 根据用户使用数据，生成专业的使用情况报告与保养建议 |
| **动态提示词** | 根据场景自动切换系统提示词，优化回答质量 |
| **流式输出** | 支持流式响应，提升用户体验 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户界面层                               │
│                      Streamlit Web App                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Agent 层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ ReAct Agent │  │  Middleware │  │   Dynamic Prompt Switch │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
│  Tools: rag_summarize | get_weather | fetch_external_data | ... │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         模型层                                   │
│  ┌─────────────┐  ┌────────────────────────────────────────────┐│
│  │ Kimi LLM    │  │ Embedding: Sentence Transformers / DashScope││
│  │ (Moonshot)  │  │                                            ││
│  └─────────────┘  └────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         存储层                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ ChromaDB    │  │ Config YAML │  │ External Data (CSV)     │  │
│  │ (向量库)    │  │ (配置文件)  │  │ (用户使用记录)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
Agent_project/
├── .env                    # 环境变量（API Key 等）
├── .env.example            # 环境变量示例
├── requirements.txt        # 依赖包（固定版本）
├── app.py                  # Streamlit 应用入口
│
├── agent/                  # Agent 模块
│   ├── react_agent.py      # ReAct Agent 定义
│   └── tools/
│       ├── agent_tools.py  # 工具函数定义
│       └── middleware.py   # 中间件（日志、提示词切换）
│
├── rag/                    # RAG 模块
│   ├── rag_service.py      # RAG 检索服务
│   └── vector_store.py     # 向量存储服务
│
├── model/                  # 模型工厂
│   └── factory.py          # ChatModel / Embedding 工厂
│
├── config/                 # 配置文件
│   ├── rag.yml             # RAG 配置
│   ├── agent.yml           # Agent 配置
│   ├── prompts.yml         # 提示词路径配置
│   └── chroma.yml          # ChromaDB 配置
│
├── prompts/                # 提示词文件
│   ├── main_prompt.txt     # 系统提示词
│   ├── rag_summarize.txt   # RAG 总结提示词
│   └── report_prompt.txt   # 报告生成提示词
│
├── data/                   # 数据目录
│   ├── extermal/           # 外部数据（用户记录）
│   └── sample/             # 示例数据
│
├── chroma_db/              # ChromaDB 向量库
├── logs/                   # 日志目录
│
├── tests/                  # 测试目录
├── scripts/                # 脚本目录
├── docs/                   # 文档目录
└── utils/                  # 工具模块
    ├── config_handler.py   # 配置加载
    ├── config_validator.py # 配置校验
    ├── prompt_loader.py    # 提示词加载
    ├── path_tool.py        # 路径工具
    └── logger_handler.py   # 日志处理
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

- Python 3.10+
- Windows / macOS / Linux

### 2. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
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
copy .env.example .env

# 编辑 .env 文件，填入真实 API Key
# KIMI_API_KEY=your_kimi_api_key_here
# DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 4. 启动应用

```bash
# 启动 Streamlit 应用
streamlit run app.py
```

访问 http://localhost:8501 即可使用。

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
chat_model_name: kimi-k2-turbo-preview
embedding_model_name: dashscope-embedding  # 或 local-embedding

kimi_base_url: https://api.moonshot.cn/v1

# 本地 Embedding 路径（可选）
embedding_local_path: "D:\\ai_models\\..."
```

### Agent 配置 (config/agent.yml)

```yaml
external_data_path: data/extermal/records.csv
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **前端** | Streamlit |
| **Agent 框架** | LangChain + LangGraph |
| **LLM** | Kimi (Moonshot) |
| **Embedding** | Sentence Transformers / DashScope |
| **向量数据库** | ChromaDB |
| **配置管理** | PyYAML + python-dotenv |

---

## 项目亮点

1. **ReAct Agent 架构**：实现思考-行动-观察闭环，自主决策工具调用
2. **动态提示词切换**：根据场景自动切换系统提示词，优化回答质量
3. **配置校验机制**：启动时 + 使用时双重校验，避免运行时错误
4. **固定依赖版本**：确保跨环境一致性，避免 API 不兼容
5. **规范目录结构**：tests/、scripts/、docs/ 分离，易于维护

---

## 后续规划

- [ ] 接入更多 LLM 后端（OpenAI、DeepSeek）
- [ ] 支持多轮对话记忆
- [ ] 添加用户反馈机制
- [ ] 部署为 Docker 容器
- [ ] 添加 API 接口

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。