"""
Memento 多策略对比测试 — 真实数据

测试维度:
  图连通性: window=3 (基线) vs window=5+KNN (改进)
  Rescaling: none / softmax τ=0.1 / power p=3 / stretch k=5

只加载模型一次，编码一次，共享向量索引。
"""

import sys, os, time, re, copy
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memento.api import Memento
from memento.models import Node
from memento.graph.memory_graph import MemoryGraph


# ─── 数据解析 ─────────────────────────────────────────────

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
            pairs.append({"question": q, "answer": a,
                          "combined": f"用户问: {q} 回答: {a[:200]}"})
    return pairs


# ─── 图构建工具 ───────────────────────────────────────────

def build_edges(memento, n_nodes, window=3, knn_k=0):
    """在已有索引的 memento 上建边"""
    node_ids = [f"mem_{i:04d}" for i in range(n_nodes)]

    # 共现边
    for i in range(0, n_nodes - (window - 1), window):
        w = node_ids[i:i + window]
        memento.activate(w)

    # KNN 边：基于向量相似度建边（用原始分数，不受 rescale 影响）
    if knn_k > 0:
        vi = memento.vector_index
        # 临时关闭 rescale，拿到原始 cosine similarity
        saved_rescale = vi._score_rescale
        vi._score_rescale = "none"
        for i in range(n_nodes):
            vec = vi.get_vector(i)
            if vec is None:
                continue
            results = vi.search(vec, k=knn_k + 1)  # +1 包含自己
            for nid, sim in results:
                if nid != node_ids[i] and sim > 0.3:
                    memento.link(node_ids[i], nid, weight=float(sim) * 0.3)
        vi._score_rescale = saved_rescale


# ─── 统计工具 ─────────────────────────────────────────────

def score_stats(scores):
    if not scores:
        return {}
    arr = np.array(scores)
    return {"min": float(arr.min()), "max": float(arr.max()),
            "spread": float(arr.max() - arr.min()),
            "std": float(arr.std()), "mean": float(arr.mean())}

def truncate(text, n=65):
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


# ─── 查询 ─────────────────────────────────────────────────

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


# ─── 主流程 ───────────────────────────────────────────────

def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")

    # ── 1. 解析数据 ──
    print(f"解析数据: {data_path}")
    pairs = parse_chat_data(data_path)
    N = len(pairs)
    print(f"解析完成: {N} 条记忆\n")

    # ── 2. 加载模型 + 编码（只做一次）──
    print(f"{'━' * 72}")
    print(f"  加载模型 + 编码 (一次性)")
    print(f"{'━' * 72}")
    t0 = time.time()

    # 用第一个实例加载模型和编码
    base = Memento(
        embedding_model=model_path, device="cuda",
        diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6,
    )
    for i, p in enumerate(pairs):
        base.add_node(p["combined"], node_id=f"mem_{i:04d}")
    base.build_index()

    # 保存编码后的向量和 id_map
    saved_index = base.vector_index.index
    saved_id_map = list(base.vector_index._id_map)
    saved_dim = base.vector_index._dimension
    encode_time = time.time() - t0
    print(f"  编码完成, 维度={saved_dim}, 耗时={encode_time:.1f}s\n")

    # ── 3. 定义测试配置 ──
    configs = [
        # (名称, window, knn_k, rescale_method, rescale_param)
        ("A: 基线(w=3, 无缩放)",           3, 0, "none",    0),
        ("B: 密图(w=5+KNN3, 无缩放)",      5, 3, "none",    0),
        ("C: 密图+power(p=3)",             5, 3, "power",   3),
        ("D: 密图+stretch(k=5)",           5, 3, "stretch", 5),
        ("E: 密图+softmax(τ=0.1)",         5, 3, "softmax", 0.1),
    ]

    # ── 4. 逐个配置构建 + 测试 ──
    all_results = {}  # config_name -> {query -> {rag, diff}}

    for cfg_name, window, knn_k, rescale, param in configs:
        print(f"{'━' * 72}")
        print(f"  构建: {cfg_name}")
        print(f"{'━' * 72}")
        t1 = time.time()

        # 创建新实例，共享已编码的 FAISS 索引
        m = Memento(
            embedding_model=model_path, device="cuda",
            diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6,
            score_temperature=param, score_rescale=rescale,
        )

        # 替换向量索引的核心数据（跳过重新编码）
        m.vector_index._index = saved_index
        m.vector_index._id_map = list(saved_id_map)
        m.vector_index._dimension = saved_dim
        m.vector_index._is_fitted = True
        # 共享模型和 tokenizer（用于编码查询）
        m.vector_index._model = base.vector_index._model
        m.vector_index._tokenizer = base.vector_index._tokenizer
        m.vector_index._torch_device = base.vector_index._torch_device
        m.vector_index._use_prefix = base.vector_index._use_prefix

        # 回填节点到图
        for i, p in enumerate(pairs):
            nid = f"mem_{i:04d}"
            vector = saved_index.reconstruct(i)
            node = Node(id=nid, text=p["combined"], vector=vector,
                        importance=0.5, vitality=1.0,
                        tags=[], source="import")
            m.graph.add_node(node)
        m._index_built = True

        # 建边
        build_edges(m, N, window=window, knn_k=knn_k)
        build_time = time.time() - t1

        n_edges = m.stats["total_edges"]
        print(f"  节点={N}, 边={n_edges}, 建图耗时={build_time:.1f}s")

        # ── 跑查询 ──
        all_results[cfg_name] = {"_edges": n_edges, "_config": cfg_name}

        for qi, query in enumerate(QUERIES, 1):
            rag = m.query_rag_only(query, k=10)
            diff = m.query(query, k=10, seed_k=20)
            all_results[cfg_name][query] = {"rag": rag, "diff": diff}

            rag_st = score_stats([it["score"] for it in rag])
            diff_st = score_stats([it["score"] for it in diff])

            print(f"  [{qi}] 「{query}」")
            print(f"    RAG:  spread={rag_st['spread']:.4f} std={rag_st['std']:.4f}"
                  f"  [{rag_st['min']:.3f}~{rag_st['max']:.3f}]")
            print(f"    扩散: spread={diff_st['spread']:.4f} std={diff_st['std']:.4f}"
                  f"  [{diff_st['min']:.3f}~{diff_st['max']:.3f}]")

            # Top-3 扩散结果
            for i, it in enumerate(diff[:3], 1):
                print(f"      {i}. (s={it['score']:.4f}) {truncate(it['text'])}")

        print()

    # ══════════════════════════════════════════════════════════
    #  5. 汇总对比表
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print(f"  汇总对比")
    print(f"{'=' * 72}")

    # 表头
    cfg_names = [c[0] for c in configs]
    header = f"  {'查询':<22s}"
    for name in cfg_names:
        short = name.split("(")[0].strip() if "(" in name else name[:8]
        header += f" | {short:>12s}"
    print(header)
    print(f"  {'-' * 22}-+-{'-' * 12}-" * len(cfg_names))

    # RAG spread
    print(f"\n  === RAG Spread (top1 - top10) ===")
    for query in QUERIES:
        q_short = query[:20]
        row = f"  {q_short:<22s}"
        for name in cfg_names:
            rag = all_results[name][query]["rag"]
            st = score_stats([it["score"] for it in rag])
            row += f" | {st['spread']:>12.4f}"
        print(row)

    # 扩散 spread
    print(f"\n  === 扩散 Spread (top1 - top10) ===")
    for query in QUERIES:
        q_short = query[:20]
        row = f"  {q_short:<22s}"
        for name in cfg_names:
            diff = all_results[name][query]["diff"]
            st = score_stats([it["score"] for it in diff])
            row += f" | {st['spread']:>12.4f}"
        print(row)

    # 平均值
    print(f"\n  === 平均 Spread ===")
    row_rag = f"  {'RAG 平均':<22s}"
    row_diff = f"  {'扩散 平均':<22s}"
    for name in cfg_names:
        rag_spreads = []
        diff_spreads = []
        for query in QUERIES:
            rag = all_results[name][query]["rag"]
            diff = all_results[name][query]["diff"]
            rag_spreads.append(score_stats([it["score"] for it in rag])["spread"])
            diff_spreads.append(score_stats([it["score"] for it in diff])["spread"])
        row_rag += f" | {np.mean(rag_spreads):>12.4f}"
        row_diff += f" | {np.mean(diff_spreads):>12.4f}"
    print(row_rag)
    print(row_diff)

    # 边数
    row_edges = f"  {'边数':<22s}"
    for name in cfg_names:
        row_edges += f" | {all_results[name]['_edges']:>12d}"
    print(f"\n{row_edges}")

    # ── 详细扩散 Top-5 对比 (选 2 个有代表性的查询) ──
    sample_queries = ["容器启动时配置丢失怎么排查", "怎么有效休息不会浪费意志力"]

    for query in sample_queries:
        print(f"\n{'━' * 72}")
        print(f"  详细对比: 「{query}」")
        print(f"{'━' * 72}")

        for name in cfg_names:
            diff = all_results[name][query]["diff"]
            print(f"\n  ┌─ {name} ─────────────────────────")
            for i, it in enumerate(diff[:5], 1):
                print(f"  │ {i}. (s={it['score']:.4f}) {truncate(it['text'])}")
            if len(diff) > 5:
                scores_rest = [it["score"] for it in diff[5:]]
                if scores_rest:
                    print(f"  │ ... ({len(diff)-5} more, max={max(scores_rest):.4f})")
            print(f"  └──────────────────────────────────────────")

    print(f"\n{'=' * 72}")
    print(f"  测试完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
