"""
惊奇度容差匹配测试

对比:
  A: raw 关键词边（无惊奇度过滤）
  B: surprisal_tolerance=0.20 的关键词边

验证惊奇度容差匹配能否减少噪声边、改善创意/情感类话题的查询质量。
"""

import sys, re, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memento.api import Memento


def parse_chat_data(filepath):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()
    parts = re.split(r'【用户提问】', content)
    pairs = []
    for part in parts:
        part = part.strip()
        if not part or '【AI 回答】' not in part:
            continue
        q, a = part.split('【AI 回答】', 1)
        q, a = q.strip(), a.strip()
        if q:
            pairs.append({"question": q, "answer": a})
    return pairs


def truncate(text, n=60):
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def score_stats(scores):
    if not scores:
        return {}
    arr = np.array(scores)
    return {"spread": float(arr.max() - arr.min()),
            "std": float(arr.std())}


QUERIES = [
    "容器启动时配置丢失怎么排查",
    "钢琴连电脑需要什么线和软件",
    "怎么提高学习效率防止晚上崩盘",
    "为什么流行歌都是情情爱爱",
    "手机传文件到电脑用什么软件",
    "独立游戏开发者要不要学美术",
    "梯子和局域网冲突怎么解决",
    "怎么有效休息不会浪费意志力",
]


def build_qa_knn(memento, pairs, knn_k=3):
    for i, p in enumerate(pairs):
        q_id, a_id = f"q_{i:04d}", f"a_{i:04d}"
        memento.add_node(p["question"], node_id=q_id, tags=["question"])
        memento.add_node(p["answer"][:500], node_id=a_id, tags=["answer"])
    memento.build_index()
    for i in range(len(pairs)):
        memento.activate([f"q_{i:04d}", f"a_{i:04d}"])
    vi = memento.vector_index
    all_ids = [f"q_{i:04d}" for i in range(len(pairs))] + \
              [f"a_{i:04d}" for i in range(len(pairs))]
    for nid in all_ids:
        node = memento.graph.get_node(nid)
        if node is None or node.vector is None:
            continue
        results = vi.search(node.vector, k=knn_k + 1)
        for cand_id, sim in results:
            if cand_id != nid and sim > 0.3:
                memento.link(nid, cand_id, weight=float(sim) * 0.3)


def run_queries(memento, queries, k=10, seed_k=20):
    results = {}
    for q in queries:
        results[q] = memento.query(q, k=k, seed_k=seed_k)
    return results


def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")
    pairs = parse_chat_data(data_path)
    print(f"数据: {len(pairs)} 条 Q&A\n")

    all_results = {}

    # ── A: raw 关键词边 ──
    print(f"{'━' * 72}")
    print(f"  A: raw 关键词边 (无惊奇度过滤)")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    res = m.build_keyword_edges(top_k=5, compute_surprisal=True)
    print(f"  边: {m.stats['total_edges']} (+{res['edges_added']} kw)")
    print(f"  惊奇度 kw 拒绝: {res['kw_surprisal_rejected']}")
    all_results["A"] = {"results": run_queries(m, QUERIES),
                        "edges": m.stats["total_edges"],
                        "kw_rejected": res["kw_surprisal_rejected"]}

    # 打印惊奇度样例
    for nid in ["q_0000", "q_0005", "a_0018"]:
        node = m.graph.get_node(nid)
        if node is None:
            continue
        surprisal = m.get_keyword_surprisal(nid)
        sorted_kws = sorted(surprisal.items(), key=lambda x: x[1], reverse=True)
        print(f"  [{nid}] {truncate(node.text, 40)}")
        for kw, s in sorted_kws:
            print(f"    {kw:<12s} {s:.4f}")

    # ── B: surprisal_tolerance=0.20 ──
    print(f"\n{'━' * 72}")
    print(f"  B: surprisal_tolerance=0.20")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    res = m.build_keyword_edges(top_k=5, surprisal_tolerance=0.20)
    print(f"  边: {m.stats['total_edges']} (+{res['edges_added']} kw)")
    print(f"  惊奇度 kw 拒绝: {res['kw_surprisal_rejected']}")
    all_results["B"] = {"results": run_queries(m, QUERIES),
                        "edges": m.stats["total_edges"],
                        "kw_rejected": res["kw_surprisal_rejected"]}

    # ── 汇总 ──
    print(f"\n\n{'=' * 72}")
    print(f"  汇总")
    print(f"{'=' * 72}")

    print(f"\n  {'查询':<24s} | {'A raw':>10s} | {'B tol=0.20':>10s}")
    print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}")

    avg_a, avg_b = [], []
    for q in QUERIES:
        da = score_stats([it["score"] for it in all_results["A"]["results"][q]])
        db = score_stats([it["score"] for it in all_results["B"]["results"][q]])
        avg_a.append(da["spread"])
        avg_b.append(db["spread"])
        diff = db["spread"] - da["spread"]
        mark = " ✓" if diff > 0.01 else (" ✗" if diff < -0.01 else "")
        print(f"  {q[:24]:<24s} | {da['spread']:>10.4f} | {db['spread']:>10.4f}{mark}")

    print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'平均':<24s} | {np.mean(avg_a):>10.4f} | {np.mean(avg_b):>10.4f}")
    print(f"  {'边数':<24s} | {all_results['A']['edges']:>10d} | {all_results['B']['edges']:>10d}")
    print(f"  {'kw拒绝':<24s} | {all_results['A']['kw_rejected']:>10d} | {all_results['B']['kw_rejected']:>10d}")

    # 详细对比
    for q in ["独立游戏开发者要不要学美术", "梯子和局域网冲突怎么解决"]:
        print(f"\n{'━' * 72}")
        print(f"  详细: 「{q}」")
        print(f"{'━' * 72}")
        for key, name in [("A", "raw"), ("B", "tol=0.20")]:
            diff = all_results[key]["results"][q]
            st = score_stats([it["score"] for it in diff])
            print(f"\n  ┌─ {name} (spread={st['spread']:.4f}) ─────")
            for i, it in enumerate(diff[:5], 1):
                tags = it.get("tags", [])
                tag_str = f" [{','.join(tags)}]" if tags else ""
                print(f"  │ {i}. (s={it['score']:.4f}){tag_str} {truncate(it['text'])}")
            print(f"  └{'─' * 60}")

    print(f"\n{'=' * 72}")
    print(f"  完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
