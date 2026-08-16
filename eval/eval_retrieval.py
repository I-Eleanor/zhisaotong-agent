"""RAG 检索能力评测：Hit Rate@K / Recall@K / MRR / 耗时 + 对照实验。

用法：
    python eval/eval_retrieval.py                    # 默认配置
    python eval/eval_retrieval.py --split dev         # 只跑开发集
    python eval/eval_retrieval.py --no-reranker       # 仅向量召回（关闭 Reranker）
    python eval/eval_retrieval.py --top-k 5           # 改变 top_k
    python eval/eval_retrieval.py --chunk-size 200    # 改变 chunk_size

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

from rag.reranker import NoopReranker, create_reranker  # noqa: E402
from rag.vector_store import VectorStoreService  # noqa: E402
from utils.config_handler import chroma_conf  # noqa: E402

DATASET_PATH = os.path.join(PROJECT_ROOT, "eval", "dataset.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval", "results")


def load_dataset(split=None):
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if split:
        data = [d for d in data if d["split"] == split]
    return data


def compute_hit_rate(results, k=3):
    hits = sum(1 for r in results if r["hit_at_k"])
    return hits / len(results) if results else 0.0


def compute_recall_at_k(results, k=3):
    if not results:
        return 0.0
    recalls = []
    for r in results:
        expected = set(r["expected_sources"])
        if not expected:
            continue
        retrieved = set(r["retrieved_sources"][:k])
        recall = len(expected & retrieved) / len(expected)
        recalls.append(recall)
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_mrr(results):
    rr_sum = 0.0
    for r in results:
        if r["first_relevant_rank"] is not None:
            rr_sum += 1.0 / r["first_relevant_rank"]
    return rr_sum / len(results) if results else 0.0


def compute_latency_stats(results):
    latencies = [r["latency_ms"] for r in results]
    if not latencies:
        return {"avg_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0}
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p95_idx = int(len(latencies) * 0.95)
    return {
        "avg_ms": round(avg, 1),
        "p95_ms": round(latencies[min(p95_idx, len(latencies) - 1)], 1),
        "min_ms": round(latencies[0], 1),
        "max_ms": round(latencies[-1], 1),
    }


def evaluate_retrieval(vs, reranker, dataset, top_k=3):
    results = []
    for item in dataset:
        if not item["expected_sources"]:
            continue
        start = time.perf_counter()
        docs = vs.get_retriever(k=top_k * 3).invoke(item["question"])
        docs = reranker.rerank(item["question"], docs)[:top_k]
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_sources = [doc.metadata.get("source_file", "") for doc in docs]
        expected = set(item["expected_sources"])

        hit_at_k = any(s in expected for s in retrieved_sources[:top_k])
        first_relevant_rank = None
        for i, s in enumerate(retrieved_sources, start=1):
            if s in expected:
                first_relevant_rank = i
                break

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_sources": list(expected),
            "retrieved_sources": retrieved_sources,
            "hit_at_k": hit_at_k,
            "first_relevant_rank": first_relevant_rank,
            "latency_ms": round(latency_ms, 1),
            "category": item["category"],
            "difficulty": item["difficulty"],
        })
    return results


def run_experiment(label, vs, reranker, dataset, top_k=3):
    print(f"\n{'='*60}")
    print(f"实验: {label}")
    print(f"{'='*60}")

    results = evaluate_retrieval(vs, reranker, dataset, top_k)

    hit_rate = compute_hit_rate(results, top_k)
    recall = compute_recall_at_k(results, top_k)
    mrr = compute_mrr(results)
    latency = compute_latency_stats(results)

    report = {
        "label": label,
        "top_k": top_k,
        "num_queries": len(results),
        "hit_rate_at_k": round(hit_rate, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "latency": latency,
        "details": results,
    }

    print(f"  查询数:     {len(results)}")
    print(f"  Hit@{top_k}:     {hit_rate:.2%}")
    print(f"  Recall@{top_k}:  {recall:.2%}")
    print(f"  MRR:        {mrr:.4f}")
    print(f"  平均耗时:   {latency['avg_ms']:.1f}ms")
    print(f"  P95耗时:    {latency['p95_ms']:.1f}ms")

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 检索能力评测")
    parser.add_argument("--split", default=None, help="只评测指定划分 (dev/test)")
    parser.add_argument("--no-reranker", action="store_true", help="关闭 Reranker")
    parser.add_argument("--top-k", type=int, default=3, help="检索 top_k")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = load_dataset(split=args.split)
    print(f"加载评测集: {len(dataset)} 条")

    vs = VectorStoreService()

    reports = []

    if not args.no_reranker:
        reranker = create_reranker(
            enabled=chroma_conf.get("reranker_enabled", False),
            model_name=chroma_conf.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k=args.top_k,
        )
        report = run_experiment("向量召回 + Reranker", vs, reranker, dataset, args.top_k)
        reports.append(report)

    noop_reranker = NoopReranker()
    report = run_experiment("仅向量召回", vs, noop_reranker, dataset, args.top_k)
    reports.append(report)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"retrieval_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")

    print(f"\n{'='*60}")
    print("对照实验汇总")
    print(f"{'='*60}")
    print(f"{'方案':<25} {'Hit@'+str(args.top_k):<12} {'MRR':<10} {'Recall@'+str(args.top_k):<12} {'平均耗时':<12}")
    print("-" * 71)
    for r in reports:
        print(f"{r['label']:<25} {r['hit_rate_at_k']:.2%}{'':<6} {r['mrr']:.4f}{'':<4} {r['recall_at_k']:.2%}{'':<6} {r['latency']['avg_ms']:.1f}ms")


if __name__ == "__main__":
    main()
