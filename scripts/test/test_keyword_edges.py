"""
关键词建边对比测试

在 Q/A 拆分 + KNN 基础上，叠加 keyatten 关键词重叠边。

对比:
  A: Q/A + KNN (615 边)
  B: Q/A + KNN + 关键词边
"""

import sys, time, re
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memento.api import Memento
from memento.index.keyatten_extractor import MemoryKeywordExtractor


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


def truncate(text, n=65):
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def score_stats(scores):
    if not scores:
        return {}
    arr = np.array(scores)
    return {"spread": float(arr.max() - arr.min()),
            "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


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
    """Q/A 拆分 + KNN 建边"""
    for i, p in enumerate(pairs):
        q_id, a_id = f"q_{i:04d}", f"a_{i:04d}"
        memento.add_node(p["question"], node_id=q_id, tags=["question"])
        memento.add_node(p["answer"][:500], node_id=a_id, tags=["answer"])
    memento.build_index()

    # Q↔A 共现边
    for i in range(len(pairs)):
        memento.activate([f"q_{i:04d}", f"a_{i:04d}"])

    # KNN 语义边
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


def add_keyword_edges(memento, pairs, top_k=5, min_overlap=2):
    """用 keyatten 提取关键词并基于重叠建边"""
    print(f"\n  ── keyatten 关键词提取 ──")
    t0 = time.time()

    ext = MemoryKeywordExtractor(device="cuda", default_top_k=top_k)

    # 收集所有节点文本，建立 IDF
    all_ids = [f"q_{i:04d}" for i in range(len(pairs))] + \
              [f"a_{i:04d}" for i in range(len(pairs))]
    all_texts = []
    for i, p in enumerate(pairs):
        all_texts.append(p["question"])
        all_texts.append(p["answer"][:500])

    vocab_size = ext.update_idf(all_texts)
    print(f"  IDF 词表: {vocab_size} 词")

    # 提取每个节点的关键词
    node_keywords = {}
    for nid, text in zip(all_ids, all_texts):
        kws = ext.extract(text)
        node_keywords[nid] = kws

    # 打印几个样例
    sample_ids = all_ids[:6]
    for nid in sample_ids:
        node = memento.graph.get_node(nid)
        kws = node_keywords[nid]
        print(f"  [{nid}] {truncate(node.text, 40)} → {kws}")

    # 用倒排索引高效建边（避免 O(n²)）
    kw_to_nodes = defaultdict(list)
    for nid, kws in node_keywords.items():
        for kw in kws:
            kw_to_nodes[kw].append(nid)

    edge_set = set()
    kw_edge_count = 0
    for kw, nids in kw_to_nodes.items():
        if len(nids) < 2 or len(nids) > 20:
            # 太少的没意义，太多的说明是常见词（跳过）
            continue
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                pair = tuple(sorted([nids[i], nids[j]]))
                if pair in edge_set:
                    continue
                edge_set.add(pair)
                # 权重 = 共享关键词数量 × 0.15
                shared = set(node_keywords[pair[0]]) & set(node_keywords[pair[1]])
                weight = len(shared) * 0.15
                memento.link(pair[0], pair[1], weight=min(weight, 0.6))
                kw_edge_count += 1

    print(f"  关键词边: {kw_edge_count} 条 (min_overlap={min_overlap})")
    print(f"  耗时: {time.time() - t0:.1f}s")
    return kw_edge_count


def run_queries(memento, queries, k=10, seed_k=20):
    results = {}
    for query in queries:
        rag = memento.query_rag_only(query, k=k)
        diff = memento.query(query, k=k, seed_k=seed_k)
        results[query] = {"rag": rag, "diff": diff}
    return results


# ─── 主流程 ───────────────────────────────────────────────

def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")

    print(f"解析数据: {data_path}")
    pairs = parse_chat_data(data_path)
    print(f"解析完成: {len(pairs)} 条 Q&A 对\n")

    # ── A: Q/A + KNN ──
    print(f"{'━' * 72}")
    print(f"  A: Q/A + KNN (无关键词边)")
    print(f"{'━' * 72}")
    m_a = Memento(
        embedding_model=model_path, device="cuda",
        diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6,
    )
    build_qa_knn(m_a, pairs, knn_k=3)
    print(f"  节点={m_a.stats['total_nodes']}, 边={m_a.stats['total_edges']}")
    res_a = run_queries(m_a, QUERIES)

    for qi, query in enumerate(QUERIES, 1):
        diff = res_a[query]["diff"]
        st = score_stats([it["score"] for it in diff])
        print(f"  [{qi}] 「{query}」 扩散spread={st['spread']:.4f}")
        for i, it in enumerate(diff[:2], 1):
            print(f"      {i}. (s={it['score']:.4f}) {truncate(it['text'])}")

    # ── B: Q/A + KNN + 关键词边 ──
    print(f"\n{'━' * 72}")
    print(f"  B: Q/A + KNN + keyatten 关键词边")
    print(f"{'━' * 72}")
    m_b = Memento(
        embedding_model=model_path, device="cuda",
        diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6,
    )
    build_qa_knn(m_b, pairs, knn_k=3)
    edges_before = m_b.stats["total_edges"]
    add_keyword_edges(m_b, pairs, top_k=5, min_overlap=2)
    edges_after = m_b.stats["total_edges"]
    print(f"\n  边: {edges_before} → {edges_after} (+{edges_after - edges_before})")

    res_b = run_queries(m_b, QUERIES)

    for qi, query in enumerate(QUERIES, 1):
        diff = res_b[query]["diff"]
        st = score_stats([it["score"] for it in diff])
        print(f"  [{qi}] 「{query}」 扩散spread={st['spread']:.4f}")
        for i, it in enumerate(diff[:2], 1):
            print(f"      {i}. (s={it['score']:.4f}) {truncate(it['text'])}")

    # ── 汇总 ──
    print(f"\n\n{'=' * 72}")
    print(f"  汇总")
    print(f"{'=' * 72}")

    avg_a, avg_b = [], []
    print(f"\n  {'查询':<24s} | {'A Q/A+KNN':>12s} | {'B +关键词':>12s}")
    print(f"  {'-'*24}-+-{'-'*12}-+-{'-'*12}")
    for query in QUERIES:
        da = score_stats([it["score"] for it in res_a[query]["diff"]])
        db = score_stats([it["score"] for it in res_b[query]["diff"]])
        avg_a.append(da["spread"])
        avg_b.append(db["spread"])
        print(f"  {query[:24]:<24s} | {da['spread']:>12.4f} | {db['spread']:>12.4f}")
    print(f"  {'-'*24}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'平均':<24s} | {np.mean(avg_a):>12.4f} | {np.mean(avg_b):>12.4f}")
    print(f"\n  边数: A={m_a.stats['total_edges']}  B={m_b.stats['total_edges']}")

    # 详细对比
    for query in ["怎么有效休息不会浪费意志力", "容器启动时配置丢失怎么排查"]:
        print(f"\n{'━' * 72}")
        print(f"  详细: 「{query}」")
        print(f"{'━' * 72}")
        for label, res, name in [("A", res_a, "Q/A+KNN"), ("B", res_b, "+关键词")]:
            diff = res[query]["diff"]
            print(f"\n  ┌─ {name} ─────────────────────────")
            for i, it in enumerate(diff[:5], 1):
                tags = it.get("tags", [])
                tag_str = f" [{','.join(tags)}]" if tags else ""
                print(f"  │ {i}. (s={it['score']:.4f}){tag_str} {truncate(it['text'])}")
            print(f"  └──────────────────────────────────────────")

    print(f"\n{'=' * 72}")
    print(f"  完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
