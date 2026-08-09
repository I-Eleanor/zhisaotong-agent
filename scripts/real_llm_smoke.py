"""真实 LLM 端到端验证（不走 mock）。

用 .venv 运行：
    ./.venv/Scripts/python.exe scripts/real_llm_smoke.py
"""
import os
import sys
import time

# 把项目根加入 sys.path（standalone 脚本运行时只把 scripts/ 加进 path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from model.factory import get_chat_model
from agent.orchestrator import Orchestrator
from agent.events import event_to_text


def test_direct_chat():
    print("=== [1] 直连 DeepSeek 对话模型 ===")
    m = get_chat_model()
    t0 = time.time()
    resp = m.invoke("用一句话介绍扫地机器人。")
    print("耗时 %.1fs" % (time.time() - t0))
    print("回复:", resp.content)
    print()


def run_query(q, mode=None):
    orch = Orchestrator()
    t0 = time.time()
    full = []
    for ev in orch.execute(q, mode=mode):
        text = event_to_text(ev)
        if text:
            full.append(text)
            print(text.rstrip())
    print("总耗时 %.1fs" % (time.time() - t0))
    print(">> 最终输出(~400字):", "".join(full)[-400:].replace("\n", " "))
    print()


if __name__ == "__main__":
    print("模型:", get_chat_model().model_name if hasattr(get_chat_model(), "model_name") else "n/a")
    test_direct_chat()
    print("=== [2] 对话 / 知识问答：怎么清理滤网 ===")
    run_query("怎么清理滤网")
    print("=== [3] 诊断（真实 LLM 驱动 Plan-Execute-Replan）：扫地机不工作了，指示灯闪红 ===")
    run_query("扫地机不工作了，指示灯一直闪红", mode="diagnostic")
    print("=== 真实 LLM 端到端验证完成 ===")
