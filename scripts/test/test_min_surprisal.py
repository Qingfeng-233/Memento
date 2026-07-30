"""
高惊奇度关键词建边测试

用 gte-small-zh (mean pooling) 计算惊奇度，只保留高惊奇度关键词建边。
高惊奇度 = 独特锚点词（如 v2rayN, 洛希），低惊奇度 = 主题内常见词。

对比:
  baseline: Q/A + KNN (无关键词边)
  A: raw 关键词边
  B: min_surprisal=0.5
  C: min_surprisal=0.6
  D: min_surprisal=0.7
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
    return {"spread": float(arr.max() - arr.min()), "std": float(arr.std())}


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

    # ── baseline ──
    print(f"{'━' * 72}")
    print(f"  baseline: Q/A + KNN")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    print(f"  边: {m.stats['total_edges']}")
    all_results["baseline"] = {"results": run_queries(m, QUERIES),
                               "edges": m.stats["total_edges"]}

    # ── A: raw ──
    print(f"\n{'━' * 72}")
    print(f"  A: raw 关键词边")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    res = m.build_keyword_edges(top_k=5, compute_surprisal=True)
    print(f"  边: {m.stats['total_edges']} (+{res['edges_added']} kw)")

    # 打印 gte 惊奇度分布
    all_s = []
    for scores in m._node_keyword_surprisal.values():
        all_s.extend(scores.values())
    arr = np.array(all_s)
    print(f"\n  gte-small-zh 惊奇度分布:")
    print(f"    均值={arr.mean():.4f}  中位={np.median(arr):.4f}  "
          f"std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 1.0)]:
        c = ((arr >= lo) & (arr < hi)).sum()
        print(f"    [{lo:.1f}-{hi:.1f})  {c:>4d} ({c/len(arr)*100:.1f}%)")

    # 样例
    for nid in ["q_0000", "q_0005", "a_0062"]:
        node = m.graph.get_node(nid)
        if node is None:
            continue
        s = m.get_keyword_surprisal(nid)
        sorted_kws = sorted(s.items(), key=lambda x: x[1], reverse=True)
        print(f"  [{nid}] {truncate(node.text, 40)}")
        for kw, v in sorted_kws:
            print(f"    {kw:<14s} {v:.4f}")

    all_results["A_raw"] = {"results": run_queries(m, QUERIES),
                            "edges": m.stats["total_edges"]}

    # ── B/C/D: min_surprisal ──
    for threshold in [0.5, 0.6, 0.7]:
        label = f"min_s={threshold}"
        print(f"\n{'━' * 72}")
        print(f"  {label}")
        print(f"{'━' * 72}")
        m = Memento(embedding_model=model_path, device="cuda",
                    diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
        build_qa_knn(m, pairs)
        res = m.build_keyword_edges(top_k=5, min_surprisal=threshold)
        print(f"  边: {m.stats['total_edges']} (+{res['edges_added']} kw)")
        print(f"  kw 拒绝: {res['kw_surprisal_rejected']}")
        all_results[label] = {"results": run_queries(m, QUERIES),
                              "edges": m.stats["total_edges"],
                              "rejected": res["kw_surprisal_rejected"]}

    # ── 汇总 ──
    print(f"\n\n{'=' * 80}")
    print(f"  汇总")
    print(f"{'=' * 80}")

    keys = ["baseline", "A_raw", "min_s=0.5", "min_s=0.6", "min_s=0.7"]
    header = f"  {'查询':<24s}"
    for k in keys:
        header += f" | {k:>10s}"
    print(header)
    print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    avgs = {k: [] for k in keys}
    for q in QUERIES:
        row = f"  {q[:24]:<24s}"
        for k in keys:
            diff = all_results[k]["results"][q]
            st = score_stats([it["score"] for it in diff])
            avgs[k].append(st["spread"])
            row += f" | {st['spread']:>10.4f}"
        print(row)

    print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    row = f"  {'平均':<24s}"
    for k in keys:
        row += f" | {np.mean(avgs[k]):>10.4f}"
    print(row)

    row = f"  {'边数':<24s}"
    for k in keys:
        row += f" | {all_results[k]['edges']:>10d}"
    print(row)

    if "rejected" in all_results.get("min_s=0.5", {}):
        row = f"  {'kw拒绝':<24s} | {'':>10s} | {'':>10s}"
        for k in ["min_s=0.5", "min_s=0.6", "min_s=0.7"]:
            row += f" | {all_results[k]['rejected']:>10d}"
        print(row)

    # 详细对比
    for q in ["独立游戏开发者要不要学美术", "手机传文件到电脑用什么软件"]:
        print(f"\n{'━' * 80}")
        print(f"  详细: 「{q}」")
        print(f"{'━' * 80}")
        for k, name in [("baseline", "baseline"), ("A_raw", "raw"),
                        ("min_s=0.5", "s≥0.5"), ("min_s=0.6", "s≥0.6")]:
            diff = all_results[k]["results"][q]
            st = score_stats([it["score"] for it in diff])
            print(f"\n  ┌─ {name} (spread={st['spread']:.4f}) ─────")
            for i, it in enumerate(diff[:5], 1):
                tags = it.get("tags", [])
                tag_str = f" [{','.join(tags)}]" if tags else ""
                print(f"  │ {i}. (s={it['score']:.4f}){tag_str} "
                      f"{truncate(it['text'])}")
            print(f"  └{'─' * 60}")

    print(f"\n{'=' * 80}")
    print(f"  完成!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
