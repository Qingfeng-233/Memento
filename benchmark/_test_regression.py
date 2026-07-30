"""Context Vector Routing 回归测试

覆盖 20 条查询，检验:
1. 消歧路由是否正确（父亲/音乐）
2. 普通查询是否退化
3. 抽象/模糊查询是否合理
4. 性能是否正常
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memento import Memento

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "testtxt.txt")

QUERIES = [
    # ── A. 消歧测试（同关键词不同语境） ──
    ("小说里主角的亲生父亲是怎么回事", "消歧-小说父亲"),
    ("帮我爸打工好累", "消歧-现实父亲"),
    ("空灵唯美的音乐推荐", "消歧-空灵音乐"),
    ("为什么流行歌都是情情爱爱", "消歧-情爱歌"),

    # ── B. 具体事实查询（应当命中明确事件） ──
    ("LocalSend 连不上怎么办", "技术-LocalSend"),
    ("v2rayN 路由怎么设置绕过局域网", "技术-VPN路由"),
    ("明月天涯那首歌叫什么", "音乐-明月天涯"),
    ("SCP基金会和三体谁厉害", "科幻-SCP三体"),

    # ── C. 小说创作相关 ──
    ("容器启动失败的原因是什么", "小说-容器配置代价"),
    ("硬魔法系统和唯心敌人怎么搭配", "小说-魔法系统"),
    ("龙族少女莉琉拉", "小说-莉琉拉"),

    # ── D. 心理/哲学 ──
    ("高敏感人群怎么应对社交疲劳", "运维-高并发"),
    ("存在主义和斯多葛学派", "哲学-存在主义"),
    ("坚持的意义是什么", "心理-坚持意义"),

    # ── E. 学习/生活 ──
    ("怎么有效休息不浪费意志力", "生活-休息"),
    ("假期作业写不完怎么办", "学习-假期作业"),
    ("独立游戏开发者要不要学美术", "游戏-学美术"),

    # ── F. 短/模糊查询 ──
    ("打游戏", "短查询-游戏"),
    ("跟我爸的关系", "短查询-关系"),

    # ── G. 长查询（多概念交叉） ──
    ("我之前聊过的关于宇宙共鸣导致变身的那个设定，后来怎么样了", "长查询-宇宙共鸣"),
]


def main():
    print("=" * 70)
    print("Context Vector Routing 回归测试")
    print("=" * 70)

    # ── 1. 加载数据 ──
    print("\n[1/4] 加载数据...")
    with open(DATA_PATH, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    entries = [e.strip() for e in raw.split("【用户提问】") if e.strip()]
    print(f"  共 {len(entries)} 条记忆")

    # ── 2. 构建系统 ──
    print("\n[2/4] 构建 Memento (Qwen3-0.6B)...")
    t0 = time.time()
    mem = Memento(
        embedding_model="models/Qwen3-Embedding-0.6B",
        diffusion_hops=2,
    )
    for entry in entries:
        mem.add_node(text=entry)
    mem.build_index()
    t_idx = time.time() - t0
    print(f"  向量索引: {t_idx:.1f}s")

    print("\n[3/4] 构建概念图 (含上下文向量)...")
    t0 = time.time()
    info = mem.build_concept_graph(
        top_k=8,
        max_concepts=300,
        min_concept_energy=0.5,
    )
    t_cg = time.time() - t0
    n_ctx = len(mem.concept_graph.context_vectors)
    print(f"  概念图: {t_cg:.1f}s")
    print(f"  概念数: {info['concepts']}, 边: {info['concept_edges']}")
    print(f"  上下文向量: {n_ctx} 条")

    # ── 3. 查询测试 ──
    print("\n[4/4] 执行 20 条查询...")
    print("=" * 70)

    all_times = []
    for query, label in QUERIES:
        t0 = time.time()
        raw = mem.query_with_concepts(query, k=5, debug=True)
        elapsed = (time.time() - t0) * 1000
        all_times.append(elapsed)

        # debug=True 返回 dict: {query, seed_concepts, activated_concepts, results}
        debug_info = raw
        results = raw.get("results", []) if isinstance(raw, dict) else raw

        print(f"\n── {label}: \"{query}\"")
        print(f"   耗时: {elapsed:.0f}ms | 结果数: {len(results)}")

        if not results:
            print("   ⚠️  无结果!")
            continue

        for i, r in enumerate(results[:3]):
            text_preview = r["text"][:80].replace("\n", " ")
            print(f"   #{i+1} score={r['score']:.3f} rag={r['rag_score']:.3f} "
                  f"concept={r['concept_score']:.3f} | {text_preview}...")

        # 显示激活的 top concepts
        activated = debug_info.get("activated_concepts", []) if isinstance(debug_info, dict) else []
        if activated:
            cstr = ", ".join(f"{c['concept']}({c['activation']:.2f})" for c in activated[:5])
            print(f"   激活概念: {cstr}")

    # ── 4. 汇总 ──
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    avg_ms = sum(all_times) / len(all_times)
    min_ms = min(all_times)
    max_ms = max(all_times)
    print(f"  查询耗时: avg={avg_ms:.0f}ms  min={min_ms:.0f}ms  max={max_ms:.0f}ms")
    print(f"  上下文向量: {n_ctx} 条")
    print(f"  概念数: {info['concepts']}")

    # 检查退化信号
    warnings = []
    if avg_ms > 200:
        warnings.append(f"⚠️  平均查询耗时偏高: {avg_ms:.0f}ms (>200ms)")
    if n_ctx == 0:
        warnings.append("⚠️  上下文向量为 0，路由功能未生效!")

    # 检查消歧是否工作
    print("\n── 消歧检查 ──")
    # 比较 "小说父亲" vs "现实父亲" 的 Top-1
    novel_idx = next(i for i, (_, l) in enumerate(QUERIES) if l == "消歧-小说父亲")
    real_idx = next(i for i, (_, l) in enumerate(QUERIES) if l == "消歧-现实父亲")

    # 重新查一次拿完整结果
    novel_res = mem.query_with_concepts("小说里主角的亲生父亲是怎么回事", k=5)
    real_res = mem.query_with_concepts("帮我爸打工好累", k=5)
    if novel_res and real_res:
        novel_top1 = novel_res[0]["text"][:60]
        real_top1 = real_res[0]["text"][:60]
        same_top1 = novel_res[0]["id"] == real_res[0]["id"]
        print(f"  小说父亲 Top1: {novel_top1}...")
        print(f"  现实父亲 Top1: {real_top1}...")
        if same_top1:
            warnings.append("⚠️  小说/现实父亲查询命中同一条结果，消歧可能失败!")
            print("  ❌ 两条查询命中相同 Top-1，消歧可能失败")
        else:
            print("  ✅ 两条查询命中不同 Top-1，消歧正常工作")

    if warnings:
        print(f"\n⚠️  发现 {len(warnings)} 个警告:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✅ 无退化信号，一切正常")


if __name__ == "__main__":
    main()
