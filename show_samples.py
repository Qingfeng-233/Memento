"""
完整样例展示 - 不截断，完整呈现查询与召回结果
"""
import json
import time
from collections import defaultdict
from pathlib import Path


def main():
    from memento.api import Memento

    # ── 初始化 ──
    data_path = Path(__file__).parent / "memories.jsonl"
    memories = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                memories.append(json.loads(line))

    mem = Memento(diffusion_hops=2)
    for m in memories:
        mem.add_node(text=m["text"], node_id=m["id"],
                     importance=m.get("importance", 0.5),
                     tags=m.get("tags", []),
                     source=m.get("source", "import"),
                     created_at=m.get("created_at"))
    mem.build_index()

    # 建图
    tag_groups = defaultdict(list)
    for m in memories:
        tags = m.get("tags", [])
        if tags:
            tag_groups[tags[0]].append(m)
    for tag, group in tag_groups.items():
        group.sort(key=lambda x: x.get("created_at", ""))
        for i in range(len(group)):
            for j in range(i + 1, min(i + 4, len(group))):
                m_a, m_b = group[i], group[j]
                n_a, n_b = mem.graph.get_node(m_a["id"]), mem.graph.get_node(m_b["id"])
                if n_a and n_b:
                    mem.graph.add_edge(m_a["id"], m_b["id"],
                                       weight=0.08 * n_a.vitality * n_b.vitality,
                                       edge_type="cooccurrence")
    sorted_mems = sorted(memories, key=lambda x: x.get("created_at", ""))
    for i in range(len(sorted_mems)):
        for j in range(i + 1, min(i + 5, len(sorted_mems))):
            m_a, m_b = sorted_mems[i], sorted_mems[j]
            shared = len(set(m_a.get("tags", [])) & set(m_b.get("tags", [])))
            mem.graph.add_edge(m_a["id"], m_b["id"],
                               weight=0.03 + 0.02 * shared,
                               edge_type="cooccurrence")

    # 跑一次睡眠让系统沉淀
    mem.trigger_sleep()

    # ════════════════════════════════════════════════════
    #  完整样例
    # ════════════════════════════════════════════════════
    samples = [
        {
            "query": "深度学习的实际应用",
            "desc": "常见主题查询，看扩散能否找到语义相关但词面不同的结果",
        },
        {
            "query": "联邦学习隐私保护协同训练",
            "desc": "精确技术概念，看RAG和扩散各表现如何",
        },
        {
            "query": "推荐系统和知识图谱有什么关系",
            "desc": "跨领域关联查询，最考验扩散能力的场景",
        },
        {
            "query": "认知科学与心理学",
            "desc": "非技术领域的交叉查询",
        },
    ]

    for idx, sample in enumerate(samples, 1):
        query = sample["query"]
        desc = sample["desc"]

        print()
        print("=" * 72)
        print(f"  样例 {idx}")
        print("=" * 72)
        print(f"  查询文本:  「{query}」")
        print(f"  测试目的:  {desc}")

        # 纯 RAG
        rag = mem.query_rag_only(query, k=5)
        # 扩散
        full = mem.query(query, k=5)

        rag_ids = {r["id"] for r in rag}
        full_ids = {r["id"] for r in full}

        print()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │              【纯 RAG 结果】(仅语义向量匹配)              │")
        print("  └─────────────────────────────────────────────────────────┘")
        for i, r in enumerate(rag, 1):
            node = mem.graph.get_node(r["id"])
            in_diff = r["id"] in full_ids
            kept = "保留" if in_diff else "被挤出"
            print()
            print(f"  #{i}  相似度={r['score']:.4f}")
            print(f"      文本: {r['text']}")
            print(f"      ID={r['id']}  标签={r['tags']}  "
                  f"重要性={r['importance']:.4f}  生命力={r['vitality']:.4f}")
            print(f"      连接数={r['edges']}  扩散后={kept}")

        print()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │        【扩散联想结果】(RAG种子 + 图扩散 + 重要性加权)     │")
        print("  └─────────────────────────────────────────────────────────┘")
        for i, r in enumerate(full, 1):
            node = mem.graph.get_node(r["id"])
            is_new = r["id"] not in rag_ids
            source_tag = "*** 扩散发现（纯RAG找不到） ***" if is_new else "(RAG种子)"

            # 获取这个节点的邻居信息
            neighbors = mem.graph.get_neighbors(r["id"])
            neighbor_count = len(neighbors)
            top_neighbors = sorted(neighbors.items(),
                                 key=lambda x: x[1].weight, reverse=True)[:3]

            print()
            print(f"  #{i}  最终得分={r['score']:.4f}  {source_tag}")
            print(f"      文本: {r['text']}")
            print(f"      ID={r['id']}  标签={r['tags']}  "
                  f"重要性={r['importance']:.4f}  生命力={r['vitality']:.4f}")
            print(f"      连接数={r['edges']}")

            if top_neighbors:
                print(f"      最强邻居:")
                for nid, edge in top_neighbors:
                    nb = mem.graph.get_node(nid)
                    if nb:
                        print(f"        -> {nid} (边权={edge.weight:.3f}, "
                              f"类型={edge.edge_type}) "
                              f"「{nb.text[:50]}」")

        # 对比小结
        overlap = rag_ids & full_ids
        new_found = full_ids - rag_ids
        print()
        print(f"  ── 小结 ──")
        print(f"  两种方法重叠: {len(overlap)} 个结果")
        print(f"  扩散新发现:   {len(new_found)} 个结果")
        if new_found:
            print(f"  新发现内容:")
            for nid in new_found:
                n = mem.graph.get_node(nid)
                if n:
                    print(f"    - 「{n.text}」")

    # ════════════════════════════════════════════════════
    #  项目状态总览
    # ════════════════════════════════════════════════════
    print()
    print()
    print("=" * 72)
    print("  项目当前状态总览")
    print("=" * 72)

    stats = mem.stats
    print(f"""
  系统规模:
    总节点:     {stats['total_nodes']}
    活跃节点:   {stats['active_nodes']}
    休眠节点:   {stats['dormant_nodes']}
    冷存储:     {stats['cold_nodes']}
    总边数:     {stats['total_edges']}
    向量维度:   {mem.vector_index.dimension}
    时钟步:     {stats['clock_step']}

  已实现功能 (对应 plan.txt 路线图):
    [x] Phase 1 核心记忆引擎
        - 节点 CRUD + TF-IDF+SVD 向量嵌入
        - 情境共现建边 (标签共现 + 时间窗口)
        - RAG 种子检索 + 多跳扩散
        - 基础时钟步衰减

    [x] Phase 2 生命力与动态系统
        - lambda(生命力) 衰减/提升/保护机制
        - omega(重要性) 手动标记 + 结构中心性沉淀
        - 2跳扩散 + omega/lambda 影响传播
        - 使用强化副作用 (查询时自动强化路径)
        - 边修剪 (w < w_min 时删除)

    [x] Phase 3 睡眠与探索
        - 回放巩固 (Top 5% 高 lambda 节点序列回放)
        - 漫游联想 (随机游走强化路径)
        - 探索边自动建立 (高 lambda 节点向量邻居探索)
        - 聚类凝聚 (贪心社区检测 + 聚合节点)
        - 遗忘与修剪 (全局衰减 + 弱边删除)

    [ ] Phase 4 对外接口与高级特性
        - 矛盾检测 (需 NLI/情感小模型)
        - 梦境报告 (需 LLM 接入)
        - 可视化图

  待优化项:
    1. 嵌入层升级: TF-IDF+SVD -> sentence-transformers
       (当前因网络无法下载模型，功能不受影响)
    2. 聚类参数调优: 当前图度数分布均匀，社区检测阈值需适配
    3. 衰减参数调大: 5个时钟步后无节点休眠，可增大 decay_rate
    4. 建边策略细化: 按 source(chat/manual/import) 差异化权重""")

    # 边类型统计
    all_edges = mem.graph.get_all_edges()
    type_counts = defaultdict(int)
    type_weights = defaultdict(list)
    for _, _, e in all_edges:
        type_counts[e.edge_type] += 1
        type_weights[e.edge_type].append(e.weight)

    print(f"  边类型详情:")
    for etype, cnt in sorted(type_counts.items()):
        ws = type_weights[etype]
        avg_w = sum(ws) / len(ws) if ws else 0
        max_w = max(ws) if ws else 0
        min_w = min(ws) if ws else 0
        print(f"    {etype:15s}: {cnt:5d} 条  "
              f"均权={avg_w:.3f}  最强={max_w:.3f}  最弱={min_w:.3f}")

    # omega 分布
    omegas = [n.importance for n in mem.graph.nodes.values()]
    omegas.sort()
    print(f"\n  重要性(omega)分布:")
    print(f"    最小={omegas[0]:.4f}  "
          f"25%={omegas[len(omegas)//4]:.4f}  "
          f"中位={omegas[len(omegas)//2]:.4f}  "
          f"75%={omegas[3*len(omegas)//4]:.4f}  "
          f"最大={omegas[-1]:.4f}")

    # lambda 分布
    lambdas = [n.vitality for n in mem.graph.nodes.values()]
    lambdas.sort()
    print(f"  生命力(lambda)分布:")
    print(f"    最小={lambdas[0]:.4f}  "
          f"25%={lambdas[len(lambdas)//4]:.4f}  "
          f"中位={lambdas[len(lambdas)//2]:.4f}  "
          f"75%={lambdas[3*len(lambdas)//4]:.4f}  "
          f"最大={lambdas[-1]:.4f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
