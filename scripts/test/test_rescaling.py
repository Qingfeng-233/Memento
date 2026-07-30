"""
Softmax 温度缩放对比测试

固定后端（Qwen3-0.6B），对比不同 temperature 下的分数分布变化。
核心指标：spread（top-bot 差值）、std（标准差）、命中率。
"""

import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "memories.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "rescaling_comparison_results.txt"

# 测试不同温度
TEMPERATURES = [0.0, 0.05, 0.1, 0.2, 0.5]

# 测试查询
TEST_QUERIES = [
    ("强化学习的实际应用", "强化学习"),
    ("知识图谱技术", "知识图谱"),
    ("推荐系统算法", "推荐系统"),
    ("如何提高编程能力", None),
    ("数据安全与隐私保护", "网络安全"),
    ("自然语言处理的最新进展", "自然语言处理"),
    ("钢琴学习入门", None),
    ("独立游戏美术设计", None),
    ("深度学习在医学影像中的应用", "深度学习"),
    ("云计算与边缘计算的区别", None),
]


def load_memories(path):
    memories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                memories.append(json.loads(line))
    return memories


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 80)
    log("  Softmax 温度缩放对比测试")
    log(f"  后端: Qwen3-0.6B | 温度: {TEMPERATURES}")
    log("=" * 80)

    # 加载数据
    memories = load_memories(DATA_PATH)
    log(f"\n  加载 {len(memories)} 条记忆")

    # 建 tag->ids 映射
    tag_to_ids = defaultdict(set)
    for m in memories:
        for t in m.get("tags", []):
            tag_to_ids[t].add(m["id"])

    # 加载模型（只加载一次）
    from memento.index.vector_index import VectorIndex

    model_path = str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B")
    # 先加载模型，temperature 后面再改
    vi = VectorIndex(model_name=model_path, device=None, score_temperature=0.0)

    node_ids = [m["id"] for m in memories]
    texts = [m["text"] for m in memories]

    t0 = time.time()
    vi.fit_and_add(node_ids, texts)
    log(f"  索引构建: {time.time()-t0:.1f}s | 维度: {vi.dimension}")

    # 对每个温度值跑一遍查询
    all_results = {}  # temp -> [query_stats]

    for temp in TEMPERATURES:
        vi._score_temperature = temp
        temp_label = f"τ={temp}" if temp > 0 else "原始"
        log(f"\n{'━' * 30} {temp_label} {'━' * 30}")

        temp_results = []
        for query, target_tag in TEST_QUERIES:
            query_vec = vi.encode([query], mode="query")[0]
            results = vi.search(query_vec, k=10)

            scores = [s for _, s in results]
            ids = [nid for nid, _ in results]

            spread = scores[0] - scores[-1] if scores else 0
            std = np.std(scores) if scores else 0

            # 命中率
            hits = None
            total = None
            if target_tag and target_tag in tag_to_ids:
                target_ids = tag_to_ids[target_tag]
                hits = sum(1 for nid in ids if nid in target_ids)
                total = len(target_ids)

            stats = {
                "query": query,
                "top": scores[0] if scores else 0,
                "bot": scores[-1] if scores else 0,
                "spread": spread,
                "std": std,
                "hits": hits,
                "total": total,
                "results": results,
            }
            temp_results.append(stats)

            hit_str = f" | 命中: {hits}/{total}" if hits is not None else ""
            log(f"  {query[:20]:<20} top={scores[0]:.4f}  bot={scores[-1]:.4f}  "
                f"spread={spread:.4f}  std={std:.4f}{hit_str}")

        all_results[temp] = temp_results

    # ═══════════════════════════════════════════════
    #  汇总对比
    # ═══════════════════════════════════════════════

    log(f"\n\n{'=' * 80}")
    log("  汇总对比")
    log("=" * 80)

    log(f"\n  {'温度':<12} {'平均 top':<12} {'平均 bot':<12} {'平均 spread':<14} "
        f"{'平均 std':<12} {'命中率':<10}")
    log(f"  {'-'*70}")

    for temp in TEMPERATURES:
        results = all_results[temp]
        tops = [r["top"] for r in results]
        bots = [r["bot"] for r in results]
        spreads = [r["spread"] for r in results]
        stds = [r["std"] for r in results]

        # 命中率
        total_hits = sum(r["hits"] for r in results if r["hits"] is not None)
        total_items = sum(r["total"] for r in results if r["hits"] is not None)
        hit_rate = total_hits / total_items if total_items > 0 else 0

        temp_label = f"τ={temp}" if temp > 0 else "原始"
        log(f"  {temp_label:<12} {np.mean(tops):<12.4f} {np.mean(bots):<12.4f} "
            f"{np.mean(spreads):<14.4f} {np.mean(stds):<12.4f} {hit_rate*100:.1f}%")

    # 展示具体查询的详细分数（选几个代表性的）
    log(f"\n\n{'=' * 80}")
    log("  代表性查询的分数分布（τ=0 vs τ=0.1）")
    log("=" * 80)

    for qi, (query, target_tag) in enumerate(TEST_QUERIES):
        raw_results = all_results[0.0][qi]["results"]
        scaled_results = all_results[0.1][qi]["results"]

        log(f"\n  Q: {query}")
        log(f"  {'#':<4} {'原始分数':<12} {'τ=0.1':<12} {'文本'}")
        log(f"  {'-'*60}")

        for i in range(min(10, len(raw_results))):
            raw_id, raw_score = raw_results[i]
            # 找到对应的 scaled score
            scaled_score = 0
            for sid, ss in scaled_results:
                if sid == raw_id:
                    scaled_score = ss
                    break
            # 找文本
            text = ""
            for m in memories:
                if m["id"] == raw_id:
                    text = m["text"][:40]
                    break
            log(f"  {i+1:<4} {raw_score:<12.4f} {scaled_score:<12.4f} {text}")

    # 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    log(f"\n  结果已保存到: {OUTPUT_PATH}")

    # 清理 GPU
    try:
        del vi
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    log("\n  测试完成!")


if __name__ == "__main__":
    main()
