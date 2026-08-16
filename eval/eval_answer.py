"""RAG 答案质量评测：规则评估 + LLM-as-Judge。

用法：
    python eval/eval_answer.py                     # 默认配置
    python eval/eval_answer.py --split dev          # 只跑开发集
    python eval/eval_answer.py --no-llm-judge       # 只做规则评估

输出 JSON 报告到 eval/results/ 目录。
"""
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from rag.rag_service import RagSummarizeService  # noqa: E402

DATASET_PATH = os.path.join(PROJECT_ROOT, "eval", "dataset.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval", "results")

JUDGE_PROMPT = """你是一个严格的RAG答案质量评估专家。请根据以下标准对答案打分。

问题：{question}
参考答案：{expected_answer}
检索上下文：{context}
系统回答：{answer}

评分标准（每项1-5分）：
1. 正确性：答案是否事实正确
2. 忠实度：答案是否得到检索上下文支持（非幻觉）
3. 完整性：答案是否完整覆盖问题的关键点
4. 引用准确率：答案中引用的来源是否准确

请以JSON格式输出：
{{"correctness": X, "faithfulness": X, "completeness": X, "citation_accuracy": X, "reasoning": "简短说明"}}
"""


def load_dataset(split=None):
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if split:
        data = [d for d in data if d["split"] == split]
    return data


def rule_evaluate(item, answer, context):
    """规则评估：关键词命中、来源匹配、拒答检测。"""
    expected_keywords = item.get("expected_keywords", [])
    expected_sources = item.get("expected_sources", [])

    keyword_hits = sum(1 for kw in expected_keywords if kw in answer)
    keyword_rate = keyword_hits / len(expected_keywords) if expected_keywords else 0.0

    source_hits = sum(1 for src in expected_sources if src in context)
    source_rate = source_hits / len(expected_sources) if expected_sources else 0.0

    no_answer_phrases = ["未检索到", "无法给出", "建议您", "知识库依据不足"]
    is_refusal = any(phrase in answer for phrase in no_answer_phrases)
    should_refuse = len(expected_sources) == 0

    refusal_correct = (is_refusal and should_refuse) or (not is_refusal and not should_refuse)

    return {
        "keyword_hit_rate": round(keyword_rate, 4),
        "keyword_hits": keyword_hits,
        "keyword_total": len(expected_keywords),
        "source_hit_rate": round(source_rate, 4),
        "source_hits": source_hits,
        "source_total": len(expected_sources),
        "is_refusal": is_refusal,
        "should_refuse": should_refuse,
        "refusal_correct": refusal_correct,
    }


def llm_judge(question, expected_answer, context, answer, model=None):
    """LLM-as-Judge：使用 DeepSeek 评判答案质量。"""
    if model is None:
        from model.factory import get_chat_model
        model = get_chat_model()

    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        context=context[:2000] if context else "(无检索上下文)",
        answer=answer,
    )

    start = time.perf_counter()
    response = model.invoke(prompt)
    latency_ms = (time.perf_counter() - start) * 1000

    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        json_start = content.index("{")
        json_end = content.rindex("}") + 1
        scores = json.loads(content[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        scores = {"correctness": 0, "faithfulness": 0, "completeness": 0, "citation_accuracy": 0, "reasoning": content[:200]}

    return {
        "scores": scores,
        "latency_ms": round(latency_ms, 1),
        "model_name": getattr(model, "model_name", getattr(model, "model", "unknown")),
    }


def evaluate_answers(rag_service, dataset, use_llm_judge=True):
    results = []
    for i, item in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {item['id']}: {item['question'][:30]}...")

        start = time.perf_counter()
        context = rag_service.build_context(item["question"])
        answer = rag_service.rag_summarize(item["question"])
        gen_latency_ms = (time.perf_counter() - start) * 1000

        rule_scores = rule_evaluate(item, answer, context)

        llm_scores = None
        if use_llm_judge and item["expected_sources"]:
            try:
                llm_scores = llm_judge(
                    item["question"],
                    item["expected_answer"],
                    context,
                    answer,
                )
            except Exception as e:
                llm_scores = {"error": str(e)}

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "actual_answer": answer,
            "context_used": context[:500] if context else "",
            "category": item["category"],
            "difficulty": item["difficulty"],
            "gen_latency_ms": round(gen_latency_ms, 1),
            "rule_evaluation": rule_scores,
            "llm_evaluation": llm_scores,
        })

    return results


def compute_summary(results):
    num = len(results)
    if num == 0:
        return {}

    avg_keyword_rate = sum(r["rule_evaluation"]["keyword_hit_rate"] for r in results) / num
    avg_source_rate = sum(r["rule_evaluation"]["source_hit_rate"] for r in results) / num
    refusal_accuracy = sum(1 for r in results if r["rule_evaluation"]["refusal_correct"]) / num
    avg_gen_latency = sum(r["gen_latency_ms"] for r in results) / num

    llm_results = [r for r in results if r.get("llm_evaluation") and "scores" in r.get("llm_evaluation", {})]
    llm_summary = {}
    if llm_results:
        for dim in ["correctness", "faithfulness", "completeness", "citation_accuracy"]:
            vals = [r["llm_evaluation"]["scores"].get(dim, 0) for r in llm_results]
            llm_summary[f"avg_{dim}"] = round(sum(vals) / len(vals), 2)

    return {
        "num_queries": num,
        "avg_keyword_hit_rate": round(avg_keyword_rate, 4),
        "avg_source_hit_rate": round(avg_source_rate, 4),
        "refusal_accuracy": round(refusal_accuracy, 4),
        "avg_gen_latency_ms": round(avg_gen_latency, 1),
        "llm_judge": llm_summary,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 答案质量评测")
    parser.add_argument("--split", default=None, help="只评测指定划分 (dev/test)")
    parser.add_argument("--no-llm-judge", action="store_true", help="关闭 LLM-as-Judge")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = load_dataset(split=args.split)
    print(f"加载评测集: {len(dataset)} 条")

    rag_service = RagSummarizeService()

    print("\n开始答案质量评测...")
    results = evaluate_answers(rag_service, dataset, use_llm_judge=not args.no_llm_judge)

    summary = compute_summary(results)

    print(f"\n{'='*60}")
    print("答案质量评测汇总")
    print(f"{'='*60}")
    print(f"  查询数:           {summary.get('num_queries', 0)}")
    print(f"  关键词命中率:     {summary.get('avg_keyword_hit_rate', 0):.2%}")
    print(f"  来源命中率:       {summary.get('avg_source_hit_rate', 0):.2%}")
    print(f"  拒答准确率:       {summary.get('refusal_accuracy', 0):.2%}")
    print(f"  平均生成耗时:     {summary.get('avg_gen_latency_ms', 0):.1f}ms")

    if summary.get("llm_judge"):
        lj = summary["llm_judge"]
        print("\n  LLM-as-Judge 评分:")
        for dim, val in lj.items():
            print(f"    {dim}: {val}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split or "all",
        "llm_judge_enabled": not args.no_llm_judge,
        "summary": summary,
        "details": results,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"answer_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
