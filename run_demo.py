"""
Memento 演示脚本

功能：
1. 加载 memories.jsonl 数据（2000 条记忆）
2. 批量构建 TF-IDF + SVD 向量索引
3. 构建情境共现图（标签共现 + 时间窗口）
4. 运行查询演示（RAG vs 扩散联想对比）
5. 运行睡眠巩固
6. 展示系统效果
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def main():
    from memento.api import Memento

    # ═══════════════════════════════════════════════════════
    #  Step 1: 初始化系统
    # ═══════════════════════════════════════════════════════
    print("=" * 60)
    print("  Memento - 双系统联想记忆架构 演示")
    print("=" * 60)
    print(f"  嵌入方案: TF-IDF + SVD (轻量级，无需下载)")
    print()

    # ═══════════════════════════════════════════════════════
    #  Step 2: 加载记忆数据
    # ═══════════════════════════════════════════════════════
    data_path = Path(__file__).parent / "memories.jsonl"
    print(f"  加载数据: {data_path}")

    memories = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                memories.append(json.loads(line))

    print(f"  数据量: {len(memories)} 条记忆")

    # ═══════════════════════════════════════════════════════
    #  Step 3: 初始化 Memento + 批量加载
    # ═══════════════════════════════════════════════════════
    t0 = time.time()
    mem = Memento(diffusion_hops=2)

    print("  导入节点...", end=" ", flush=True)
    for m in memories:
        mem.add_node(
            text=m["text"],
            node_id=m["id"],
            importance=m.get("importance", 0.5),
            tags=m.get("tags", []),
            source=m.get("source", "import"),
            created_at=m.get("created_at"),
        )

    print("构建向量索引...", end=" ", flush=True)
    count = mem.build_index()
    print(f"完成! ({count} 个节点, {time.time() - t0:.1f}s)")
    print(f"  向量维度: {mem.vector_index.dimension}")
    print()

    # ═══════════════════════════════════════════════════════
    #  Step 4: 构建情境共现图
    # ═══════════════════════════════════════════════════════
    print("  构建情境共现图...")

    # 4a. 基于主题标签的共现
    tag_groups = defaultdict(list)
    for m in memories:
        tags = m.get("tags", [])
        if tags:
            tag_groups[tags[0]].append(m)

    edges_created = 0
    for tag, group in tag_groups.items():
        group.sort(key=lambda x: x.get("created_at", ""))
        window_size = 4
        for i in range(len(group)):
            for j in range(i + 1, min(i + window_size, len(group))):
                m_a, m_b = group[i], group[j]
                n_a = mem.graph.get_node(m_a["id"])
                n_b = mem.graph.get_node(m_b["id"])
                if n_a and n_b:
                    delta_w = 0.08 * n_a.vitality * n_b.vitality
                    mem.graph.add_edge(m_a["id"], m_b["id"],
                                       weight=delta_w,
                                       edge_type="cooccurrence")
                    edges_created += 1

    print(f"    主题共现边: {edges_created}")

    # 4b. 基于时间窗口的共现
    sorted_mems = sorted(memories, key=lambda x: x.get("created_at", ""))
    time_edges = 0
    window_size = 5
    for i in range(len(sorted_mems)):
        for j in range(i + 1, min(i + window_size, len(sorted_mems))):
            m_a, m_b = sorted_mems[i], sorted_mems[j]
            tags_a = set(m_a.get("tags", []))
            tags_b = set(m_b.get("tags", []))
            shared = len(tags_a & tags_b)
            base_w = 0.03 + 0.02 * shared
            mem.graph.add_edge(m_a["id"], m_b["id"],
                               weight=base_w,
                               edge_type="cooccurrence")
            time_edges += 1

    print(f"    时间窗口边: {time_edges}")
    print(f"    总边数: {mem.graph.edge_count}")
    print()

    # ═══════════════════════════════════════════════════════
    #  Step 5: 系统初始状态
    # ═══════════════════════════════════════════════════════
    stats = mem.stats
    print("  初始状态:")
    print(f"    节点总数:  {stats['total_nodes']}")
    print(f"    活跃节点:  {stats['active_nodes']}")
    print(f"    边总数:    {stats['total_edges']}")
    print()

    # 高度数节点
    top_degree = mem.graph.get_nodes_sorted(
        lambda n: len(mem.graph.get_neighbors(n.id)))[:8]
    print("  连接度最高的节点:")
    for i, node in enumerate(top_degree, 1):
        deg = len(mem.graph.get_neighbors(node.id))
        print(f"    {i}. [{node.id}] (度={deg}, w={node.importance:.3f}) "
              f"{node.text[:50]}")
    print()

    # ═══════════════════════════════════════════════════════
    #  Step 6: 查询演示 - RAG vs 扩散联想
    # ═══════════════════════════════════════════════════════
    print("=" * 60)
    print("  查询演示：纯 RAG vs 扩散联想")
    print("=" * 60)

    queries = [
        "深度学习模型压缩与优化",
        "自然语言处理的应用",
        "推荐系统与知识图谱",
        "如何学习人工智能",
        "联邦学习隐私保护",
    ]

    for q_idx, query in enumerate(queries, 1):
        print(f"\n  -- 查询 {q_idx}: \"{query}\" --")

        rag_results = mem.query_rag_only(query, k=5)
        full_results = mem.query(query, k=5)

        print(f"\n  {'[纯 RAG]':^50}")
        for i, r in enumerate(rag_results, 1):
            print(f"    {i}. (sim={r['score']:.4f}) {r['text'][:55]}")

        print(f"\n  {'[扩散联想]':^50}")
        for i, r in enumerate(full_results, 1):
            tag = ""
            if not any(rr["id"] == r["id"] for rr in rag_results):
                tag = "  * 扩散发现"
            print(f"    {i}. (s={r['score']:.4f}, w={r['importance']:.2f}, "
                  f"v={r['vitality']:.2f}) {r['text'][:45]}{tag}")

    # ═══════════════════════════════════════════════════════
    #  Step 7: 情境激活演示
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  情境激活演示")
    print("=" * 60)
    print("  模拟：用户同时想到「深度学习」和「自然语言处理」")

    dl_nodes = [m["id"] for m in memories
                if "深度学习" in m.get("tags", [])][:3]
    nlp_nodes = [m["id"] for m in memories
                 if "自然语言处理" in m.get("tags", [])][:3]

    activation_set = dl_nodes + nlp_nodes
    if len(activation_set) >= 2:
        mem.activate(activation_set)
        print(f"  激活了 {len(activation_set)} 个节点")

        if len(activation_set) >= 2:
            edge = mem.graph.get_edge(activation_set[0], activation_set[1])
            if edge:
                print(f"  边 {activation_set[0]} <-> {activation_set[1]}: "
                      f"w={edge.weight:.4f}")

    # ═══════════════════════════════════════════════════════
    #  Step 8: 主动关联演示
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  主动关联演示")
    print("=" * 60)
    print("  模拟：用户认为「人工智能」和「哲学思想」有深层联系")

    ai_nodes = [m["id"] for m in memories
                if "人工智能" in m.get("tags", [])][:1]
    phil_nodes = [m["id"] for m in memories
                  if "哲学思想" in m.get("tags", [])][:1]

    if ai_nodes and phil_nodes:
        mem.link(ai_nodes[0], phil_nodes[0], weight=0.8)
        edge = mem.graph.get_edge(ai_nodes[0], phil_nodes[0])
        print(f"  已建立手动关联: {ai_nodes[0]} <-> {phil_nodes[0]}")
        print(f"  边类型: {edge.edge_type}, 强度: {edge.weight:.4f}")

    # ═══════════════════════════════════════════════════════
    #  Step 9: 第一次睡眠
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  触发第一次睡眠周期...")
    print("=" * 60)

    t0 = time.time()
    report = mem.trigger_sleep()
    print(report.summary())
    print(f"  睡眠耗时: {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════
    #  Step 10: 睡眠后查询对比
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  睡眠后查询对比")
    print("=" * 60)

    test_query = "深度学习模型压缩与优化"
    print(f"\n  查询: \"{test_query}\"")

    post_sleep = mem.query(test_query, k=5)
    print(f"\n  睡眠后扩散联想结果:")
    for i, r in enumerate(post_sleep, 1):
        print(f"    {i}. (s={r['score']:.4f}, w={r['importance']:.2f}, "
              f"v={r['vitality']:.2f}, edges={r['edges']}) {r['text'][:45]}")

    # ═══════════════════════════════════════════════════════
    #  Step 11: 衰减 + 第二次睡眠
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  推进 5 个时钟步 + 第二次睡眠...")
    print("=" * 60)

    for _ in range(5):
        mem.clock_step()

    stats2 = mem.stats
    print(f"  5 个时钟步后:")
    print(f"    活跃: {stats2['active_nodes']}, "
          f"休眠: {stats2['dormant_nodes']}, "
          f"冷存储: {stats2['cold_nodes']}")

    report2 = mem.trigger_sleep()
    print(report2.summary())

    # ═══════════════════════════════════════════════════════
    #  Step 12: 最终状态
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  最终系统状态")
    print("=" * 60)

    final_stats = mem.stats
    for k, v in final_stats.items():
        print(f"    {k}: {v}")

    # Top 重要节点
    top_omega = mem.graph.get_nodes_sorted(
        lambda n: n.importance)[:5]
    print("\n  重要性最高的节点 (Top 5):")
    for i, n in enumerate(top_omega, 1):
        print(f"    {i}. [{n.id}] w={n.importance:.4f} v={n.vitality:.4f} "
              f"edges={n.edge_count} | {n.text[:45]}")

    # Top 活跃节点
    top_lambda = mem.graph.get_nodes_sorted(
        lambda n: n.vitality)[:5]
    print("\n  生命力最高的节点 (Top 5):")
    for i, n in enumerate(top_lambda, 1):
        print(f"    {i}. [{n.id}] v={n.vitality:.4f} w={n.importance:.4f} "
              f"edges={n.edge_count} | {n.text[:45]}")

    # 聚类节点
    clusters = [n for n in mem.graph.nodes.values()
                if "__cluster__" in n.tags]
    if clusters:
        print(f"\n  聚合节点: {len(clusters)} 个")
        for c in clusters[:3]:
            print(f"    [{c.id}] w={c.importance:.3f} | {c.text[:70]}")

    # 边类型统计
    all_edges = mem.graph.get_all_edges()
    type_counts = defaultdict(int)
    for _, _, e in all_edges:
        type_counts[e.edge_type] += 1
    print(f"\n  边类型分布:")
    for etype, cnt in sorted(type_counts.items()):
        print(f"    {etype}: {cnt}")

    # ═══════════════════════════════════════════════════════
    #  Step 13: 保存
    # ═══════════════════════════════════════════════════════
    save_dir = Path(__file__).parent / "saved_memory"
    print(f"\n  保存记忆系统到: {save_dir}")
    mem.save(str(save_dir))
    print("  保存完成!")

    print("\n" + "=" * 60)
    print("  Memento 演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
