"""
surprisal_top_k 建边测试

对比配置：
  A. Q/A + KNN 基线（无关键词边）
  B. 全量关键词边（top_k=5，无惊奇度筛选）
  C. surprisal_top_k=3（每节点只留惊奇度最高的 3 个词建边）
  D. surprisal_top_k=2（每节点只留惊奇度最高的 2 个词建边）
  E. surprisal_top_k=1（每节点只留惊奇度最高的 1 个词建边）

指标：边数、平均扩散度
"""

import sys, re
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


def make_base(data_pairs, model_path):
    """创建基础 Memento（Q/A 节点 + FAISS + Q↔A 共现 + KNN 语义边）"""
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    for i, p in enumerate(data_pairs):
        q_id, a_id = f"q_{i:04d}", f"a_{i:04d}"
        m.add_node(p["question"], node_id=q_id, tags=["question"])
        m.add_node(p["answer"][:500], node_id=a_id, tags=["answer"])
    m.build_index()

    # Q↔A 共现边
    for i in range(len(data_pairs)):
        m.activate([f"q_{i:04d}", f"a_{i:04d}"])

    # KNN 语义边
    vi = m.vector_index
    all_ids = [f"q_{i:04d}" for i in range(len(data_pairs))] + \
              [f"a_{i:04d}" for i in range(len(data_pairs))]
    for nid in all_ids:
        node = m.graph.get_node(nid)
        if node is None or node.vector is None:
            continue
        results = vi.search(node.vector, k=4)
        for cand_id, sim in results:
            if cand_id != nid and sim > 0.3:
                m.link(nid, cand_id, weight=float(sim) * 0.3)

    return m


def eval_diffusion(m, queries, label):
    """用一组查询测扩散效果"""
    spreads = []
    for q in queries:
        results = m.query(q, k=10, seed_k=20)
        if results:
            scores = [r["score"] for r in results]
            spread = np.std(scores) / (np.mean(scores) + 1e-9)
            spreads.append(spread)
    avg = np.mean(spreads) if spreads else 0
    all_edges = m.graph.get_all_edges()
    n_edges = len(all_edges)
    kw_edges = sum(1 for _, _, e in all_edges if e.edge_type == "keyword")
    knn_edges = n_edges - kw_edges
    print(f"  [{label}]")
    print(f"    总边数: {n_edges} (KNN={knn_edges}, 关键词={kw_edges})")
    print(f"    平均扩散度: {avg:.4f}  (std={np.std(spreads):.4f})")
    return {"label": label, "n_edges": n_edges, "kw_edges": kw_edges,
            "avg_spread": avg, "std_spread": np.std(spreads)}


# ── 测试用查询（覆盖多话题）──
QUERIES = [
    "宇宙文明 黑暗森林",
    "分布式系统分析",
    "v2rayN 代理设置",
    "钢琴学习方法",
    "三体 歌者",
    "南京 生活经历",
    "魔法系统设计",
    "艺术 审美",
    "手机文件传输",
    "学习效率 方法",
]


def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")
    pairs = parse_chat_data(data_path)

    print(f"{'=' * 70}")
    print(f"  surprisal_top_k 建边对比测试")
    print(f"{'=' * 70}")

    results = []

    # A: 基线
    print(f"\n── A: Q/A + KNN 基线 ──")
    m = make_base(pairs, model_path)
    results.append(eval_diffusion(m, QUERIES, "A: baseline"))

    # B: 全量关键词
    print(f"\n── B: 全量关键词边 (top_k=5) ──")
    m = make_base(pairs, model_path)
    res = m.build_keyword_edges(top_k=5)
    print(f"    建边报告: edges={res['edges_added']}, "
          f"total_kw={res['total_keywords']}, vocab={res['vocab_size']}")
    results.append(eval_diffusion(m, QUERIES, "B: all keywords"))

    # C: surprisal_top_k=3
    print(f"\n── C: surprisal_top_k=3 ──")
    m = make_base(pairs, model_path)
    res = m.build_keyword_edges(top_k=5, surprisal_top_k=3)
    print(f"    建边报告: edges={res['edges_added']}, "
          f"topk_rejected={res['kw_topk_rejected']}, "
          f"total_kw={res['total_keywords']}")
    results.append(eval_diffusion(m, QUERIES, "C: top_k=3"))

    # D: surprisal_top_k=2
    print(f"\n── D: surprisal_top_k=2 ──")
    m = make_base(pairs, model_path)
    res = m.build_keyword_edges(top_k=5, surprisal_top_k=2)
    print(f"    建边报告: edges={res['edges_added']}, "
          f"topk_rejected={res['kw_topk_rejected']}, "
          f"total_kw={res['total_keywords']}")
    results.append(eval_diffusion(m, QUERIES, "D: top_k=2"))

    # E: surprisal_top_k=1
    print(f"\n── E: surprisal_top_k=1 ──")
    m = make_base(pairs, model_path)
    res = m.build_keyword_edges(top_k=5, surprisal_top_k=1)
    print(f"    建边报告: edges={res['edges_added']}, "
          f"topk_rejected={res['kw_topk_rejected']}, "
          f"total_kw={res['total_keywords']}")
    results.append(eval_diffusion(m, QUERIES, "E: top_k=1"))

    # ── 汇总 ──
    print(f"\n\n{'=' * 70}")
    print(f"  汇总")
    print(f"{'=' * 70}")
    print(f"  {'配置':<22s} | {'总边数':>6s} | {'关键词边':>6s} | {'平均扩散':>8s}")
    print(f"  {'-'*22}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for r in results:
        print(f"  {r['label']:<22s} | {r['n_edges']:>6d} | {r['kw_edges']:>6d} "
              f"| {r['avg_spread']:>8.4f}")

    print(f"\n{'=' * 70}")
    print(f"  完成!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
