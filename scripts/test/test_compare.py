"""
对比测试：纯 RAG vs 扩散联想

测试多种场景，清晰展示两种检索方式的差异
"""

import json
import time
from collections import defaultdict
from pathlib import Path


def print_divider(title=""):
    if title:
        print(f"\n{'─'*25} {title} {'─'*25}")
    else:
        print("─" * 70)


def print_result_row(idx, r, is_new=False, width=50):
    """打印一行结果"""
    tag = "  >>> 扩散发现" if is_new else ""
    text = r["text"][:width]
    print(f"  {idx}. [s={r['score']:.3f} | w={r['importance']:.2f} | "
          f"v={r['vitality']:.2f} | e={r['edges']:2d}] {text}{tag}")


def run_comparison(mem, query, k=8, rag_ids=None):
    """执行一组对比查询"""
    rag = mem.query_rag_only(query, k=k)
    full = mem.query(query, k=k)

    rag_id_set = {r["id"] for r in rag}
    full_id_set = {r["id"] for r in full}

    # 统计
    overlap = rag_id_set & full_id_set
    only_rag = rag_id_set - full_id_set
    only_diff = full_id_set - rag_id_set

    print(f"\n  纯 RAG Top-{k}:")
    for i, r in enumerate(rag, 1):
        in_diff = r["id"] in full_id_set
        marker = "" if in_diff else "  (被扩散挤出)"
        print(f"    {i}. [sim={r['score']:.3f}] {r['text'][:50]}{marker}")

    print(f"\n  扩散联想 Top-{k}:")
    for i, r in enumerate(full, 1):
        is_new = r["id"] not in rag_id_set
        print_result_row(i, r, is_new=is_new)

    print(f"\n  对比统计:")
    print(f"    重叠: {len(overlap)} | 仅RAG: {len(only_rag)} | 扩散新增: {len(only_diff)}")

    return rag, full


def main():
    from memento.api import Memento

    print("=" * 70)
    print("  Memento 对比测试：纯 RAG vs 扩散联想")
    print("=" * 70)

    # 加载数据
    data_path = Path(__file__).parent / "memories.jsonl"
    memories = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                memories.append(json.loads(line))

    # 初始化
    t0 = time.time()
    mem = Memento(diffusion_hops=2)
    for m in memories:
        mem.add_node(text=m["text"], node_id=m["id"],
                     importance=m.get("importance", 0.5),
                     tags=m.get("tags", []),
                     source=m.get("source", "import"),
                     created_at=m.get("created_at"))
    mem.build_index()

    # 构建共现图
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

    print(f"  系统就绪: {mem.graph.node_count} 节点, "
          f"{mem.graph.edge_count} 边 ({time.time()-t0:.1f}s)")

    # ═════════════════════════════════════════════════════
    #  测试 1: 精确语义匹配
    # ═════════════════════════════════════════════════════
    print_divider("测试 1: 精确语义 — 「强化学习的应用」")
    run_comparison(mem, "强化学习的应用", k=8)

    # ═════════════════════════════════════════════════════
    #  测试 2: 跨领域模糊查询
    # ═════════════════════════════════════════════════════
    print_divider("测试 2: 跨领域模糊 — 「如何提高学习效率」")
    run_comparison(mem, "如何提高学习效率", k=8)

    # ═════════════════════════════════════════════════════
    #  测试 3: 罕见概念
    # ═════════════════════════════════════════════════════
    print_divider("测试 3: 罕见概念 — 「量子计算与边缘计算」")
    run_comparison(mem, "量子计算与边缘计算", k=8)

    # ═════════════════════════════════════════════════════
    #  测试 4: 高重要性节点能否被优先召回
    # ═════════════════════════════════════════════════════
    print_divider("测试 4: 重要性加权 — 「网络安全学习」")
    print("  (高 w 节点应在扩散结果中排名更靠前)")
    run_comparison(mem, "网络安全学习", k=8)

    # ═════════════════════════════════════════════════════
    #  测试 5: 激活后再查询 — 模拟「刚想过 A，再查 B」
    # ═════════════════════════════════════════════════════
    print_divider("测试 5: 情境激活后查询")

    print("\n  Step A: 用户最近一直在想「知识图谱」和「推荐系统」...")
    kg_nodes = [m["id"] for m in memories if "知识图谱" in m.get("tags", [])][:5]
    rs_nodes = [m["id"] for m in memories if "推荐系统" in m.get("tags", [])][:5]
    mem.activate(kg_nodes + rs_nodes)
    print(f"  激活了 {len(kg_nodes + rs_nodes)} 个节点 (知识图谱 + 推荐系统)")

    print("\n  Step B: 然后查「数据结构的书」...")
    print("  (预期: 与知识图谱/推荐系统关联的节点会因激活扩散而排前)")
    run_comparison(mem, "数据结构的书", k=8)

    # ═════════════════════════════════════════════════════
    #  测试 6: 睡眠前后对比
    # ═════════════════════════════════════════════════════
    print_divider("测试 6: 睡眠前后对比 — 「机器翻译」")

    print("\n  [睡眠前]")
    pre_sleep = mem.query("机器翻译", k=8)
    for i, r in enumerate(pre_sleep, 1):
        print(f"    {i}. [s={r['score']:.3f} w={r['importance']:.2f} "
              f"v={r['vitality']:.2f}] {r['text'][:50]}")

    print("\n  ... 执行睡眠周期 ...")
    report = mem.trigger_sleep()
    print(f"  回放 {report.replay_count} 节点, 强化 {report.edges_strengthened} 边, "
          f"探索 {report.explore_edges_created} 新边")

    print("\n  [睡眠后]")
    post_sleep = mem.query("机器翻译", k=8)
    for i, r in enumerate(post_sleep, 1):
        is_new = not any(p["id"] == r["id"] for p in pre_sleep)
        tag = "  >>> 新出现" if is_new else ""
        print(f"    {i}. [s={r['score']:.3f} w={r['importance']:.2f} "
              f"v={r['vitality']:.2f}] {r['text'][:50]}{tag}")

    # 得分变化
    pre_scores = {r["id"]: r["score"] for r in pre_sleep}
    post_scores = {r["id"]: r["score"] for r in post_sleep}
    changes = []
    for nid in set(pre_scores) | set(post_scores):
        pre_s = pre_scores.get(nid, 0)
        post_s = post_scores.get(nid, 0)
        if pre_s > 0:
            changes.append((nid, pre_s, post_s, post_s - pre_s))
    changes.sort(key=lambda x: x[3], reverse=True)
    if changes:
        print(f"\n  得分变化最大的节点:")
        for nid, pre_s, post_s, delta in changes[:5]:
            node = mem.graph.get_node(nid)
            arrow = "+" if delta >= 0 else ""
            print(f"    {nid}: {pre_s:.3f} -> {post_s:.3f} ({arrow}{delta:.3f}) "
                  f"| {node.text[:40] if node else ''}")

    # ═════════════════════════════════════════════════════
    #  测试 7: 手动关联的效果
    # ═════════════════════════════════════════════════════
    print_divider("测试 7: 手动关联效果")

    print("\n  建立关联: 「音乐理论」<->「健康养生」 (w=0.8)")
    music = [m["id"] for m in memories if "音乐理论" in m.get("tags", [])][:2]
    health = [m["id"] for m in memories if "健康养生" in m.get("tags", [])][:2]
    if music and health:
        mem.link(music[0], health[0], weight=0.8)
        print(f"  {music[0]} <-> {health[0]}")

    print("\n  查询「养生方法」, 看是否能联想到音乐:")
    run_comparison(mem, "养生方法", k=8)

    print("\n" + "=" * 70)
    print("  对比测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
