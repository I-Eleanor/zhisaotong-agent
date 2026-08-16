"""pytest 公共 fixture。

设计要点：
- 通过打补丁 `ChatModelFactory.generator` / `EmbeddingsFactory.generator`，让所有
  经由 `get_chat_model()` / `get_embed_model()`（含各模块里的别名导入）拿到的都是
  确定性假模型，测试完全不依赖真实 LLM / Embedding，零 API 费用、零网络。
- 假 Chat 模型按 system prompt 内容返回不同脚本化回复，足以驱动诊断 Agent 的
  plan / replan / report 节点走完完整流程。
- API 测试在路由层把 `get_orchestrator` 替换成返回固定事件流的桩，避免构造真实
  ConversationAgent（其需要支持 bind_tools 的模型）。
"""
import hashlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------- 假 Embedding
class FakeEmbeddings:
    """确定性 Embedding：同一文本永远得到同一向量，无需任何模型权重。"""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _vec(self, text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float((b % 100) / 50.0 - 1.0) for b in h[: self.dim]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


# --------------------------------------------------------------------- 假 Chat 模型
class _AIMessage:
    def __init__(self, content):
        self.content = content


class ScriptedChatModel:
    """按输入（system prompt 或纯 prompt 文本）返回脚本化回复的假模型。

    只实现诊断 / 摘要 / 路由用到的 `invoke(messages)` 接口：
    - planner prompt → 返回含「设备状态」类步骤的 JSON 计划（避免触发 RAG）
    - replan prompt → 返回 {"action": "end"}，让诊断流程尽快进入报告
    - report prompt → 返回 Markdown 诊断报告
    - 分类 / 摘要 prompt → 返回普通文本
    """

    def invoke(self, messages, *args, **kwargs):
        text = messages if isinstance(messages, str) else (messages[0].content if messages else "")
        s = text.lower()

        # 顺序很重要：replan 必须在 report 之前。
        # replan 的 system 提示同时含「重规划」与「报告」（"...进入报告生成"），
        # 若先判 report 会误把 replan 当成报告、返回 Markdown 导致 JSON 解析失败。
        # 用 replan 专属的「重规划」/「replan」做精确判定，避免误匹配报告提示。
        if "重规划" in text or "replan" in s:
            return _AIMessage('{"action": "end", "reason": "信息已足够"}')
        if "报告" in text or "report" in s or "markdown" in s:
            return _AIMessage(
                "# 诊断报告\n## 故障描述\n测试故障\n## 故障原因\n示例原因\n"
                "## 处置建议\n示例处置建议\n## 参考资料\n- 故障排除.txt"
            )
        if "计划" in text or "json" in s or "排查" in text:
            return _AIMessage('{"plan": ["查询设备运行状态", "查询耗材状态"]}')
        if "意图" in text or "classif" in s or "conversation" in s:
            return _AIMessage("conversation")
        return _AIMessage("这是一条测试回复。")


# --------------------------------------------------------------------- 桩 Orchestrator
class CannedOrchestrator:
    """API 测试用：返回固定事件流，完全绕开真实 Agent 与模型。"""

    def execute(self, query, history=None, mode=None):
        yield {"type": "tool_start", "agent": "conversation",
               "content": "", "data": {"tool": "rag_summarize", "args": {}}}
        yield {"type": "message", "agent": "conversation", "content": "这是测试回复。"}
        yield {"type": "done", "agent": "conversation", "content": ""}

    async def aexecute(self, query, history=None, mode=None):
        for ev in self.execute(query, history, mode):
            yield ev


# --------------------------------------------------------------------- 自动打桩模型
@pytest.fixture(autouse=True)
def mock_models(monkeypatch):
    """把所有模型工厂重定向到假模型（含各模块里的别名导入）。"""
    import model.factory as mf

    monkeypatch.setattr(mf.ChatModelFactory, "generator",
                        lambda self=None: ScriptedChatModel())
    monkeypatch.setattr(mf.EmbeddingsFactory, "generator",
                        lambda self=None: FakeEmbeddings())
    mf.reset_models()
    yield
    mf.reset_models()


# --------------------------------------------------------------------- 临时向量库
@pytest.fixture
def temp_vector_store(tmp_path):
    from langchain_core.documents import Document

    from rag.vector_store import VectorStoreService

    vs = VectorStoreService(
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )
    doc = Document(
        page_content="扫地机器人滤网建议每三个月更换一次，平时用软布擦拭。",
        metadata={"source_file": "滤网.txt", "chunk_index": 0, "page": -1},
    )
    vs.vector_store.add_documents([doc])
    return vs


# --------------------------------------------------------------------- FastAPI 测试客户端
@pytest.fixture
def api_client(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    fake = CannedOrchestrator()
    monkeypatch.setattr("api.routes.conversation.get_orchestrator", lambda: fake)
    monkeypatch.setattr("api.routes.diagnostic.get_orchestrator", lambda: fake)
    return TestClient(app)
