"""
惊奇度计算验证测试

验证 build_keyword_edges(compute_surprisal=True) 功能：
  - 关键词全部保留，不做过滤
  - 每个关键词附加惊奇度分数 (1 - cos_sim)
  - 观察技术类 vs 创意类话题的惊奇度分布差异
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


def truncate(text, n=50):
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")

    print(f"解析数据: {data_path}")
    pairs = parse_chat_data(data_path)
    print(f"解析完成: {len(pairs)} 条 Q&A 对\n")

    # 构建 Memento
    print(f"{'━' * 72}")
    print(f"  构建图 + 关键词 + 惊奇度")
    print(f"{'━' * 72}")

    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)

    # Q/A 拆分 + KNN
    for i, p in enumerate(pairs):
        q_id, a_id = f"q_{i:04d}", f"a_{i:04d}"
        m.add_node(p["question"], node_id=q_id, tags=["question"])
        m.add_node(p["answer"][:500], node_id=a_id, tags=["answer"])
    m.build_index()

    for i in range(len(pairs)):
        m.activate([f"q_{i:04d}", f"a_{i:04d}"])

    vi = m.vector_index
    all_ids = [f"q_{i:04d}" for i in range(len(pairs))] + \
              [f"a_{i:04d}" for i in range(len(pairs))]
    knn_edges = 0
    for nid in all_ids:
        node = m.graph.get_node(nid)
        if node is None or node.vector is None:
            continue
        results = vi.search(node.vector, k=4)
        for cand_id, sim in results:
            if cand_id != nid and sim > 0.3:
                m.link(nid, cand_id, weight=float(sim) * 0.3)
                knn_edges += 1
    print(f"  Q/A + KNN 完成: {m.stats['total_nodes']} 节点, "
          f"{m.stats['total_edges']} 边")

    # 关键词建边 + 惊奇度
    t0 = time.time()
    result = m.build_keyword_edges(top_k=5, compute_surprisal=True)
    elapsed = time.time() - t0

    print(f"\n  关键词建边完成:")
    print(f"    边: {m.stats['total_edges']} (+{result['edges_added']} 关键词边)")
    print(f"    关键词: {result['total_keywords']} 个 "
          f"(IDF 词表 {result['vocab_size']})")
    print(f"    耗时: {elapsed:.1f}s")

    # ── 惊奇度样例 ──
    print(f"\n{'━' * 72}")
    print(f"  惊奇度样例 (surprisal = 1 - cos(kw, text))")
    print(f"{'━' * 72}")

    # 挑几个有代表性的节点
    sample_ids = [
        ("q_0000", "小说/宇宙设定"),
        ("q_0005", "三体/论战"),
        ("a_0018", "钢琴设备"),
        ("a_0062", "v2rayN/梯子"),
        ("q_0040", "学习效率"),
        ("q_0054", "独立游戏美术"),
    ]

    all_surprisals = []

    for nid, label in sample_ids:
        node = m.graph.get_node(nid)
        if node is None:
            continue
        kws = m.get_node_keywords(nid)
        surprisal = m.get_keyword_surprisal(nid)

        print(f"\n  [{nid}] {label}")
        print(f"  文本: {truncate(node.text, 60)}")
        print(f"  关键词 + 惊奇度:")

        # 按惊奇度排序
        sorted_kws = sorted(surprisal.items(), key=lambda x: x[1], reverse=True)
        for kw, score in sorted_kws:
            all_surprisals.append(score)
            bar = "█" * int(score * 20)
            print(f"    {kw:<12s}  {score:.4f}  {bar}")

    # ── 惊奇度分布统计 ──
    print(f"\n{'━' * 72}")
    print(f"  惊奇度分布统计")
    print(f"{'━' * 72}")

    arr = np.array(all_surprisals)
    print(f"  总关键词数: {len(arr)}")
    print(f"  均值: {arr.mean():.4f}")
    print(f"  中位数: {np.median(arr):.4f}")
    print(f"  标准差: {arr.std():.4f}")
    print(f"  最小: {arr.min():.4f}")
    print(f"  最大: {arr.max():.4f}")

    # 分桶
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    hist, _ = np.histogram(arr, bins=bins)
    print(f"\n  分布:")
    for i in range(len(hist)):
        lo, hi = bins[i], bins[i + 1]
        count = hist[i]
        pct = count / len(arr) * 100
        bar = "█" * int(pct / 2)
        print(f"    [{lo:.1f}-{hi:.1f})  {count:>4d} ({pct:>5.1f}%)  {bar}")

    # ── 查询验证 ──
    QUERIES = [
        "梯子和局域网冲突怎么解决",
        "独立游戏开发者要不要学美术",
        "怎么有效休息不会浪费意志力",
    ]

    print(f"\n{'━' * 72}")
    print(f"  查询验证 (扩散结果)")
    print(f"{'━' * 72}")

    for query in QUERIES:
        diff = m.query(query, k=10, seed_k=20)
        scores = [it["score"] for it in diff]
        spread = max(scores) - min(scores) if scores else 0
        print(f"\n  「{query}」  spread={spread:.4f}")
        for i, it in enumerate(diff[:3], 1):
            tags = it.get("tags", [])
            tag_str = f" [{','.join(tags)}]" if tags else ""
            print(f"    {i}. (s={it['score']:.4f}){tag_str} "
                  f"{truncate(it['text'], 55)}")

    print(f"\n{'=' * 72}")
    print(f"  完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
