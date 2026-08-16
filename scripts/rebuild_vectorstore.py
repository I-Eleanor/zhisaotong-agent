import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from rag.vector_store import VectorStoreService  # noqa: E402

vs = VectorStoreService()
vs.load_document()

retriever = vs.get_retriever()
res = retriever.invoke("扫地机器人维护")
for r in res:
    print(f"内容: {r.page_content[:80]}...")
    print(f"元数据: {r.metadata}")
    print("-" * 40)

print("向量库重建完成！")
