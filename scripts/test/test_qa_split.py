"""
Memento 建边策略对比 — Q&A 拆分窗口版

核心改动：每条聊天记录拆成「用户提问」和「AI回答」两个节点，
每个完整 Q&A 对作为一个 activate 窗口建边。

对比:
  A: 旧方案（合并文本, window=3 硬切）
  B: 新方案（Q/A 拆分, 每对 Q&A activate）
  C: 新方案 + KNN=3 语义建边
"""

import sys, time, re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memento.api import Memento
from memento.models import Node


# ─── 数据解析 ─────────────────────────────────────────────

def parse_chat_data(filepath):
    """解析 testtxt.txt，返回 Q&A 对列表"""
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


def truncate(text, n=65):
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def score_stats(scores):
    if not scores:
        return {}
    arr = np.array(scores)
    return {"min": float(arr.min()), "max": float(arr.max()),
            "spread": float(arr.max() - arr.min()),
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


# ─── 方案 A：旧方案（合并文本, window=3）──────────────────

def build_old_style(memento, pairs):
    """旧方案：每条 Q&A 合并成一条记忆，window=3 硬切建边"""
    for i, p in enumerate(pairs):
        combined = f"用户问: {p['question']} 回答: {p['answer'][:200]}"
        memento.add_node(combined, node_id=f"mem_{i:04d}")
    memento.build_index()

    node_ids = [f"mem_{i:04d}" for i in range(len(pairs))]
    for i in range(0, len(node_ids) - 2, 3):
        memento.activate(node_ids[i:i + 3])


# ─── 方案 B/C：新方案（Q/A 拆分, 每对 activate）──────────

def build_qa_split(memento, pairs, knn_k=0):
    """新方案：每条 Q&A 拆成 Q 节点 + A 节点，每对作为一个 activate 窗口"""
    for i, p in enumerate(pairs):
        q_id = f"q_{i:04d}"
        a_id = f"a_{i:04d}"
        # 回答截断到 500 字（保留核心内容，避免超长）
        answer_text = p["answer"][:500]
        memento.add_node(p["question"], node_id=q_id, tags=["question"])
        memento.add_node(answer_text, node_id=a_id, tags=["answer"])
    memento.build_index()

    # 每对 Q&A 作为一个 activate 窗口
    for i in range(len(pairs)):
        q_id = f"q_{i:04d}"
        a_id = f"a_{i:04d}"
        memento.activate([q_id, a_id])

    # 可选：KNN 语义建边
    if knn_k > 0:
        vi = memento.vector_index
        all_ids = [f"q_{i:04d}" for i in range(len(pairs))] + \
                  [f"a_{i:04d}" for i in range(len(pairs))]
        for nid in all_ids:
            vec = memento.graph.get_node(nid)
            if vec is None or vec.vector is None:
                continue
            results = vi.search(vec.vector, k=knn_k + 1)
            for cand_id, sim in results:
                if cand_id != nid and sim > 0.3:
                    memento.link(nid, cand_id, weight=float(sim) * 0.3)


# ─── 测试运行器 ──────────────────────────────────────────

def run_queries(memento, queries, k=10, seed_k=20):
    """跑所有查询，返回 {query: {rag, diff}}"""
    results = {}
    for query in queries:
        rag = memento.query_rag_only(query, k=k)
        diff = memento.query(query, k=k, seed_k=seed_k)
        results[query] = {"rag": rag, "diff": diff}
    return results


def print_query_results(name, results, queries):
    """打印单个配置的查询结果"""
    for qi, query in enumerate(queries, 1):
        rag = results[query]["rag"]
        diff = results[query]["diff"]
        rag_st = score_stats([it["score"] for it in rag])
        diff_st = score_stats([it["score"] for it in diff])

        print(f"  [{qi}] 「{query}」")
        print(f"    RAG:  spread={rag_st['spread']:.4f}  [{rag_st['min']:.3f}~{rag_st['max']:.3f}]")
        print(f"    扩散: spread={diff_st['spread']:.4f}  [{diff_st['min']:.3f}~{diff_st['max']:.3f}]")
        for i, it in enumerate(diff[:3], 1):
            print(f"      {i}. (s={it['score']:.4f}) {truncate(it['text'])}")


# ─── 主流程 ──────────────────────────────────────────────

def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")

    # 解析数据
    print(f"解析数据: {data_path}")
    pairs = parse_chat_data(data_path)
    print(f"解析完成: {len(pairs)} 条 Q&A 对\n")

    # ── 方案 A：旧方案 ──
    print(f"{'━' * 72}")
    print(f"  A: 旧方案（合并文本, window=3 硬切）")
    print(f"{'━' * 72}")
    t0 = time.time()
    m_a = Memento(
        embedding_model=model_path, device="cuda",
        diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6,
    )
    build_old_style(m_a, pairs)
    print(f"  节点={m_a.stats['total_nodes']}, 边={m_a.stats['total_edges']},"
          f" 耗时={time.time()-t0:.1f}s\n")

    results_a = run_queries(m_a, QUERIES)
    print_query_results("A", results_a, QUERIES)

    # ── 方案 B：Q&A 拆分 ──
    print(f"\n{'━' * 72}")
    print(f"  B: 新方案（Q/A 拆分, 每对 activate）")
    print(f"{'━' * 72}")
    t0 = time.time()
    m_b = Memento(
        embedding_model=model_path, device="cuda",
        diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6,
    )
    build_qa_split(m_b, pairs, knn_k=0)
    print(f"  节点={m_b.stats['total_nodes']}, 边={m_b.stats['total_edges']},"
          f" 耗时={time.time()-t0:.1f}s\n")

    results_b = run_queries(m_b, QUERIES)
    print_query_results("B", results_b, QUERIES)

    # ── 方案 C：Q&A 拆分 + KNN ──
    print(f"\n{'━' * 72}")
    print(f"  C: 新方案 + KNN=3 语义建边")
    print(f"{'━' * 72}")
    t0 = time.time()
    m_c = Memento(
        embedding_model=model_path, device="cuda",
        diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6,
    )
    build_qa_split(m_c, pairs, knn_k=3)
    print(f"  节点={m_c.stats['total_nodes']}, 边={m_c.stats['total_edges']},"
          f" 耗时={time.time()-t0:.1f}s\n")

    results_c = run_queries(m_c, QUERIES)
    print_query_results("C", results_c, QUERIES)

    # ══════════════════════════════════════════════════════════
    #  汇总
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 72}")
    print(f"  汇总对比")
    print(f"{'=' * 72}")

    all_res = {"A": results_a, "B": results_b, "C": results_c}
    all_edges = {
        "A": m_a.stats["total_edges"],
        "B": m_b.stats["total_edges"],
        "C": m_c.stats["total_edges"],
    }
    all_nodes = {
        "A": m_a.stats["total_nodes"],
        "B": m_b.stats["total_nodes"],
        "C": m_c.stats["total_nodes"],
    }

    # 逐查询 spread 对比
    print(f"\n  === 扩散 Spread ===")
    print(f"  {'查询':<24s} | {'A 旧方案':>10s} | {'B Q/A拆分':>10s} | {'C +KNN':>10s}")
    print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    avg_a, avg_b, avg_c = [], [], []
    for query in QUERIES:
        da = score_stats([it["score"] for it in all_res["A"][query]["diff"]])
        db = score_stats([it["score"] for it in all_res["B"][query]["diff"]])
        dc = score_stats([it["score"] for it in all_res["C"][query]["diff"]])
        avg_a.append(da["spread"])
        avg_b.append(db["spread"])
        avg_c.append(dc["spread"])
        print(f"  {query[:24]:<24s} | {da['spread']:>10.4f} | {db['spread']:>10.4f} | {dc['spread']:>10.4f}")

    print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'平均':<24s} | {np.mean(avg_a):>10.4f} | {np.mean(avg_b):>10.4f} | {np.mean(avg_c):>10.4f}")

    print(f"\n  节点数: A={all_nodes['A']}  B={all_nodes['B']}  C={all_nodes['C']}")
    print(f"  边数:   A={all_edges['A']}  B={all_edges['B']}  C={all_edges['C']}")

    # 详细对比：选 2 个查询
    for query in ["容器启动时配置丢失怎么排查", "怎么有效休息不会浪费意志力"]:
        print(f"\n{'━' * 72}")
        print(f"  详细: 「{query}」")
        print(f"{'━' * 72}")
        for label in ["A", "B", "C"]:
            diff = all_res[label][query]["diff"]
            cfg_name = {"A": "旧方案(w=3)", "B": "Q/A拆分", "C": "Q/A+KNN"}[label]
            print(f"\n  ┌─ {cfg_name} ─────────────────────────")
            for i, it in enumerate(diff[:5], 1):
                tags = it.get("tags", [])
                tag_str = f" [{','.join(tags)}]" if tags else ""
                print(f"  │ {i}. (s={it['score']:.4f}){tag_str} {truncate(it['text'])}")
            print(f"  └──────────────────────────────────────────")

    print(f"\n{'=' * 72}")
    print(f"  测试完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
