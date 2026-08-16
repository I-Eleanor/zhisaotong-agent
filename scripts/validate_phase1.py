"""阶段一运行期验证（对照 upgrade-plan.html 第5章验收标准）。

测试项：
  T1 路由：关键词命中 -> diagnostic；无关键词 -> LLM 兜底 conversation
  T2 多轮记忆：先问滤网清理，再问「那多久换一次」应能指代滤网
  T3 诊断报告：输入清洁效率问题 -> 最终 report 事件含「原因」+「建议」
  T4 重排器：rag_service 按 chroma.yml 加载 CrossEncoder（失败则记录降级）

用法：PYTHONPATH=<项目根> .venv/Scripts/python.exe scripts/validate_phase1.py
"""
import sys

from dotenv import load_dotenv

load_dotenv(".env")

from agent.diagnostic_agent import DiagnosticAgent  # noqa: E402
from agent.memory.conversation_buffer import ConversationBuffer  # noqa: E402
from agent.orchestrator import MODE_CONVERSATION, MODE_DIAGNOSTIC, Orchestrator  # noqa: E402

results = {}

# ---------------- T1 路由 ----------------
print("=" * 60)
print("T1 路由测试")
orch = Orchestrator()
r1 = orch.route("扫地机不工作了")
r2 = orch.route("怎么清理滤网")
print(f"  route('扫地机不工作了') = {r1}  (期望 diagnostic)")
print(f"  route('怎么清理滤网')   = {r2}  (期望 conversation)")
results["T1"] = (r1 == MODE_DIAGNOSTIC) and (r2 == MODE_CONVERSATION)

# ---------------- T2 多轮记忆 ----------------
print("=" * 60)
print("T2 多轮记忆测试")
buf = ConversationBuffer()
conv = orch.conversation_agent
q1 = "扫地机器人怎么清理滤网"
evs1 = list(conv.stream(q1, history=buf.get_history_for_query()))
a1 = "".join(e.get("content", "") for e in evs1 if e.get("type") == "message")
buf.add_user_message(q1)
buf.add_assistant_message(a1)
q2 = "那多久换一次"
evs2 = list(conv.stream(q2, history=buf.get_history_for_query()))
a2 = "".join(e.get("content", "") for e in evs2 if e.get("type") == "message")
print(f"  第一轮回复片段: {a1[:60]}...")
print(f"  第二轮问题: {q2}")
print(f"  第二轮回复: {a2[:120]}")
# 判定：第二轮回复应提及滤网/滤芯相关
hit = any(k in a2 for k in ("滤网", "滤芯", "过滤器", "filter"))
results["T2"] = hit
print(f"  指代命中(滤网/滤芯): {hit}")

# ---------------- T3 诊断报告 ----------------
print("=" * 60)
print("T3 诊断报告测试")
report_text = ""
for ev in DiagnosticAgent().run("我的扫地机器人最近清洁效率很低"):
    if ev.get("type") == "report":
        report_text = ev.get("content", "")
    elif ev.get("type") == "error":
        print("  ERROR:", ev.get("content"))
if report_text:
    print(f"  报告长度: {len(report_text)}")
    print("  报告前 300 字:\n", report_text[:300])
    has_cause = any(k in report_text for k in ("故障原因", "原因", "排查", "分析"))
    has_suggest = any(k in report_text for k in ("建议", "处置", "措施", "处理", "方案"))
    results["T3"] = has_cause and has_suggest
    print(f"  含原因:{has_cause} 含建议:{has_suggest}")
else:
    results["T3"] = False
    print("  未生成 report 事件")

# ---------------- T4 重排器 ----------------
print("=" * 60)
print("T4 重排器加载测试")
try:
    from rag.rag_service import RagSummarizeService
    svc = RagSummarizeService()
    print(f"  reranker 类型: {type(svc.reranker).__name__}")
    results["T4"] = type(svc.reranker).__name__ != "NoopReranker"
except Exception as e:
    print("  reranker 初始化异常:", type(e).__name__, str(e)[:200])
    results["T4"] = False

# ---------------- 汇总 ----------------
print("=" * 60)
print("汇总:")
for k, v in results.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
overall = all(results.values())
print("OVERALL:", "ALL_PASS" if overall else "HAS_FAIL")
sys.exit(0 if overall else 1)
