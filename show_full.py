"""
三大板块完整展示:
  1. 搜索功能 — 输入查询文本, 返回排序结果
  2. 召回功能 — 激活一组记忆, 看系统联想到什么
  3. 项目状态 — 当前实现了什么, 数据什么样
"""
import json
import time
from collections import defaultdict
from pathlib import Path


def build_system():
    """构建并返回 Memento 系统"""
    from memento.api import Memento

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

    # 建共现图
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

    # 跑一次睡眠
    mem.trigger_sleep()
    return mem, memories


# ═══════════════════════════════════════════════════════════════
#  板块 1: 搜索功能
# ═══════════════════════════════════════════════════════════════
def demo_search(mem):
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  板块 1: 搜索功能".center(62) + "        ║")
    print("║" + "  输入自然语言查询 → 系统返回最相关的记忆(按得分排序)".center(48) + "║")
    print("╚" + "═" * 68 + "╝")

    queries = [
        "深度学习的模型压缩和优化技术",
        "自然语言处理和机器翻译",
        "如何系统地学习人工智能",
    ]

    for qi, query in enumerate(queries, 1):
        results = mem.query(query, k=6)

        print()
        print(f"  {'━' * 60}")
        print(f"  查询 {qi}: 「{query}」")
        print(f"  {'━' * 60}")
        print()

        for rank, r in enumerate(results, 1):
            node = mem.graph.get_node(r["id"])
            # 获取该节点的 top 邻居
            neighbors = mem.graph.get_neighbors(r["id"])
            top_nb = sorted(neighbors.items(),
                            key=lambda x: x[1].weight, reverse=True)[:2]

            print(f"  ┌─ 第 {rank} 名 ─────────────────────────────────────────")
            print(f"  │ 得分: {r['score']:.4f}")
            print(f"  │ 文本: {r['text']}")
            print(f"  │ ID:   {r['id']}")
            print(f"  │ 标签: {r['tags']}")
            print(f"  │ 重要性(omega):   {r['importance']:.4f}")
            print(f"  │ 生命力(lambda):  {r['vitality']:.4f}")
            print(f"  │ 连接数:          {r['edges']}")
            if top_nb:
                print(f"  │ 主要关联记忆:")
                for nid, edge in top_nb:
                    nb = mem.graph.get_node(nid)
                    if nb:
                        print(f"  │   └─ {nid} (边权={edge.weight:.3f}) "
                              f"「{nb.text}」")
            print(f"  └──────────────────────────────────────────────────")
            print()


# ═══════════════════════════════════════════════════════════════
#  板块 2: 召回功能
# ═══════════════════════════════════════════════════════════════
def demo_recall(mem, memories):
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  板块 2: 召回功能 (联想记忆)".center(58) + "        ║")
    print("║" + "  激活一组记忆 → 能量沿图边扩散 → 系统「联想」到更多相关记忆".center(40) + "║")
    print("╚" + "═" * 68 + "╝")

    # ── 场景 A: 情境共现 → 再查相关话题 ──
    print()
    print(f"  {'━' * 60}")
    print(f"  场景 A: 用户最近在想「强化学习」和「推荐系统」")
    print(f"  {'━' * 60}")

    rl = [m["id"] for m in memories if "强化学习" in m.get("tags", [])][:4]
    rs = [m["id"] for m in memories if "推荐系统" in m.get("tags", [])][:4]
    activated = rl + rs

    print(f"\n  被激活的记忆 ({len(activated)} 条):")
    for nid in activated:
        n = mem.graph.get_node(nid)
        if n:
            print(f"    - [{nid}] w={n.importance:.3f} | 「{n.text}」")

    mem.activate(activated)
    print(f"\n  系统操作:")
    print(f"    - 这 {len(activated)} 条记忆之间两两建立了共现边")
    print(f"    - 所有被激活记忆的生命力(lambda)提升")

    # 验证边
    sample_edge = mem.graph.get_edge(activated[0], activated[-1])
    if sample_edge:
        print(f"    - 示例边: {activated[0]} <-> {activated[-1]}  "
              f"权={sample_edge.weight:.3f}  类型={sample_edge.edge_type}")

    # 激活后查「数据结构的书」
    print(f"\n  然后用户查询: 「机器学习的书」")
    print(f"  预期: 刚激活的强化学习/推荐系统相关记忆会因联想而排名靠前")
    print()

    results = mem.query("机器学习的书", k=6)
    for rank, r in enumerate(results, 1):
        was_activated = r["id"] in activated
        tag = " <<< 刚刚被激活的记忆" if was_activated else ""
        print(f"  ┌─ 第 {rank} 名 ─────────────────────────────────────────")
        print(f"  │ 得分: {r['score']:.4f}")
        print(f"  │ 文本: {r['text']}")
        print(f"  │ ID:   {r['id']}  标签: {r['tags']}{tag}")
        print(f"  │ 重要性: {r['importance']:.4f}  生命力: {r['vitality']:.4f}  "
              f"连接数: {r['edges']}")
        print(f"  └──────────────────────────────────────────────────")
        print()

    # ── 场景 B: 手动关联 → 跨界联想 ──
    print(f"  {'━' * 60}")
    print(f"  场景 B: 用户主动建立跨领域关联")
    print(f"  {'━' * 60}")

    music = [m["id"] for m in memories if "音乐理论" in m.get("tags", [])][:1]
    health = [m["id"] for m in memories if "运动健身" in m.get("tags", [])][:1]

    if music and health:
        print(f"\n  用户操作: 我认为「音乐理论」和「运动健身」有深层联系")
        print(f"  调用 link(music, health, w=0.8)")

        mem.link(music[0], health[0], weight=0.8)

        mn = mem.graph.get_node(music[0])
        hn = mem.graph.get_node(health[0])
        edge = mem.graph.get_edge(music[0], health[0])
        print(f"\n  关联已建立:")
        print(f"    节点A: [{music[0]}] 「{mn.text}」")
        print(f"    节点B: [{health[0]}] 「{hn.text}」")
        print(f"    边:    权={edge.weight:.2f}  类型={edge.edge_type}")

        # 先激活音乐节点，再查运动
        print(f"\n  激活音乐理论的 3 条记忆...")
        music_more = [m["id"] for m in memories
                      if "音乐理论" in m.get("tags", [])][:3]
        mem.activate(music_more)
        for mid in music_more:
            n = mem.graph.get_node(mid)
            print(f"    - [{mid}] 「{n.text}」")

        print(f"\n  然后查询: 「健身锻炼方法」")
        print()
        results = mem.query("健身锻炼方法", k=6)
        for rank, r in enumerate(results, 1):
            is_music = "音乐理论" in r.get("tags", [])
            tag = " <<< 音乐领域的记忆! (跨界联想)" if is_music else ""
            print(f"  ┌─ 第 {rank} 名 ─────────────────────────────────────────")
            print(f"  │ 得分: {r['score']:.4f}")
            print(f"  │ 文本: {r['text']}")
            print(f"  │ ID:   {r['id']}  标签: {r['tags']}{tag}")
            print(f"  │ 重要性: {r['importance']:.4f}  生命力: {r['vitality']:.4f}")
            print(f"  └──────────────────────────────────────────────────")
            print()


# ═══════════════════════════════════════════════════════════════
#  板块 3: 项目状态
# ═══════════════════════════════════════════════════════════════
def demo_status(mem):
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  板块 3: 项目当前状态".center(60) + "        ║")
    print("╚" + "═" * 68 + "╝")

    stats = mem.stats

    print(f"""
  ┌─ 系统规模 ──────────────────────────────────────────────
  │ 节点总数:       {stats['total_nodes']}
  │   活跃:         {stats['active_nodes']}
  │   休眠:         {stats['dormant_nodes']}
  │   冷存储:       {stats['cold_nodes']}
  │ 边总数:         {stats['total_edges']}
  │ 向量维度:       {mem.vector_index.dimension}
  │ 当前时钟步:     {stats['clock_step']}
  └─────────────────────────────────────────────────────────""")

    # omega 分布
    omegas = sorted([n.importance for n in mem.graph.nodes.values()])
    lambdas = sorted([n.vitality for n in mem.graph.nodes.values()])
    print(f"""
  ┌─ 属性分布 ──────────────────────────────────────────────
  │ 重要性(omega)  最小={omegas[0]:.4f}  "
     "中位={omegas[len(omegas)//2]:.4f}  最大={omegas[-1]:.4f}
  │ 生命力(lambda) 最小={lambdas[0]:.4f}  "
     "中位={lambdas[len(lambdas)//2]:.4f}  最大={lambdas[-1]:.4f}
  └─────────────────────────────────────────────────────────""")

    # 边类型
    all_edges = mem.graph.get_all_edges()
    tc = defaultdict(int)
    tw = defaultdict(list)
    for _, _, e in all_edges:
        tc[e.edge_type] += 1
        tw[e.edge_type].append(e.weight)
    print(f"""
  ┌─ 边类型统计 ────────────────────────────────────────────""")
    for t, c in sorted(tc.items()):
        ws = tw[t]
        print(f"  │ {t:15s}  {c:5d} 条  "
              f"均权={sum(ws)/len(ws):.3f}  "
              f"最强={max(ws):.3f}  最弱={min(ws):.3f}")
    print(f"  └─────────────────────────────────────────────────────────")

    # 路线图
    print(f"""
  ┌─ 路线图完成度 (对应 plan.txt) ──────────────────────────
  │
  │ [x] Phase 1: 核心记忆引擎
  │     节点CRUD, TF-IDF+SVD向量嵌入, FAISS索引
  │     情境共现建边 (标签共现 + 时间窗口)
  │     RAG种子检索 + 2跳激活扩散
  │     基础时钟步衰减
  │
  │ [x] Phase 2: 生命力与动态系统
  │     lambda 衰减/提升/保护 (高omega节点有下限保护)
  │     omega 手动标记 + 结构中心性自动沉淀
  │     扩散中 omega/lambda 影响传播强度
  │     使用强化副作用 (查询时自动强化命中路径)
  │     边修剪 (w < 0.01 时删除)
  │
  │ [x] Phase 3: 睡眠与探索
  │     回放巩固 (Top 5% 高lambda序列回放, 强化相邻边)
  │     漫游联想 (10次随机游走, 按边权加权选择)
  │     探索建边 (高lambda节点自动探索向量邻居, kappa=0.05)
  │     聚类凝聚 (贪心社区检测 + 聚合节点)
  │     遗忘修剪 (全局衰减 + 弱边清除)
  │
  │ [ ] Phase 4: 对外接口与高级特性
  │     [ ] 矛盾检测 (需NLI/情感分类小模型)
  │     [ ] 梦境报告 (需LLM接入)
  │     [ ] 图可视化
  │
  └─────────────────────────────────────────────────────────""")

    # 文件结构
    print(f"""
  ┌─ 项目文件 ──────────────────────────────────────────────
  │ memento/
  │   models.py              Node/Edge 数据模型
  │   api.py                 Memento 主接口 (增删查改/睡眠/持久化)
  │   index/vector_index.py  TF-IDF + SVD + FAISS 向量索引
  │   graph/memory_graph.py  记忆图 (节点+邻接表+边管理)
  │   engine/
  │     diffusion.py         激活扩散引擎 (种子→多跳传播→强化)
  │     decay.py             时间衰减 (lambda/omega/w 动态)
  │     sleep.py             睡眠巩固 (回放/漫游/探索/聚类/修剪)
  │ run_demo.py              完整演示 (加载→建图→查询→睡眠)
  │ test_compare.py          RAG vs 扩散对比测试 (7组)
  │ show_samples.py          完整样例展示 (本脚本)
  │ memories.jsonl           2000条测试数据
  │ saved_memory/            持久化输出 (nodes/edges/vectors)
  └─────────────────────────────────────────────────────────""")

    # 待优化
    print(f"""
  ┌─ 待优化 ────────────────────────────────────────────────
  │ 1. 嵌入层: TF-IDF+SVD → sentence-transformers
  │    (当前因HuggingFace网络超时用了轻量替代, 功能不受影响)
  │ 2. 聚类: 当前图度数分布均匀, 社区检测阈值需针对性调参
  │ 3. 衰减: decay_rate偏保守, 可增大以看到休眠/冷存储效果
  │ 4. 建边: 可按source(chat/manual/import)差异化共现权重
  └─────────────────────────────────────────────────────────""")


def main():
    print()
    print("  " + "=" * 66)
    print("  " + "  Memento 双系统联想记忆 — 完整展示".center(54))
    print("  " + "=" * 66)

    mem, memories = build_system()
    print(f"\n  系统就绪: {mem.graph.node_count} 节点, "
          f"{mem.graph.edge_count} 边")

    demo_search(mem)
    demo_recall(mem, memories)
    demo_status(mem)

    print("\n  展示完毕。\n")


if __name__ == "__main__":
    main()
