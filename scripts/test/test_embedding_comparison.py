"""
嵌入模型对比测试：TF-IDF vs Qwen3-0.6B vs Qwen3-4B

核心指标：
  1. 分数分布（min/max/std/spread） — 越宽越好，说明模型能区分相关与不相关
  2. Top-K 结果质量 — 是否命中正确标签
  3. 编码速度 — 实际部署可行性
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "memories.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "embedding_comparison_results.txt"

# 三个后端配置
BACKENDS = {
    "TF-IDF": {
        "model_name": "tfidf-svd",
        "device": None,
    },
    "Qwen3-0.6B": {
        "model_name": str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B"),
        "device": None,  # auto-detect GPU
    },
    "Qwen3-4B": {
        "model_name": str(PROJECT_ROOT / "models" / "Qwen" / "Qwen3-Embedding-4B"),
        "device": None,  # auto-detect GPU
    },
}

# 测试查询：覆盖精确匹配、模糊语义、跨领域
TEST_QUERIES = [
    # 精确语义（数据中有对应标签）
    ("强化学习的实际应用", "强化学习"),
    ("知识图谱技术", "知识图谱"),
    ("推荐系统算法", "推荐系统"),
    # 模糊 / 跨领域
    ("如何提高编程能力", None),
    ("数据安全与隐私保护", "网络安全"),
    ("自然语言处理的最新进展", "自然语言处理"),
    # 边缘测试
    ("钢琴学习入门", None),         # 数据中没有音乐相关——应该全部低分
    ("独立游戏美术设计", None),      # 同上
    ("深度学习在医学影像中的应用", "深度学习"),
    ("云计算与边缘计算的区别", None),
]


# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def load_memories(path):
    """加载 memories.jsonl"""
    memories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                memories.append(json.loads(line))
    return memories


def build_index_direct(model_name, device, memories):
    """直接用 VectorIndex 构建索引，不走 Memento 全栈"""
    from memento.index.vector_index import VectorIndex

    vi = VectorIndex(model_name=model_name, device=device)
    node_ids = [m["id"] for m in memories]
    texts = [m["text"] for m in memories]

    t0 = time.time()
    vi.fit_and_add(node_ids, texts)
    build_time = time.time() - t0

    return vi, build_time


def run_query(vi, query, k=10, target_tag=None):
    """执行查询并收集统计"""
    t0 = time.time()
    query_vec = vi.encode([query], mode="query")[0]
    encode_time = time.time() - t0

    t0 = time.time()
    results = vi.search(query_vec, k=k)
    search_time = time.time() - t0

    scores = [score for _, score in results]
    ids = [nid for nid, _ in results]

    stats = {
        "query": query,
        "target_tag": target_tag,
        "encode_time_ms": round(encode_time * 1000, 1),
        "search_time_ms": round(search_time * 1000, 2),
        "top_score": round(scores[0], 4) if scores else 0,
        "bot_score": round(scores[-1], 4) if scores else 0,
        "score_spread": round(scores[0] - scores[-1], 4) if scores else 0,
        "score_mean": round(np.mean(scores), 4) if scores else 0,
        "score_std": round(np.std(scores), 4) if scores else 0,
        "results": results,
    }

    # 如果有目标标签，检查命中率
    if target_tag:
        stats["target_hits"] = 0
        # 需要外部传入 memories_map

    return stats


# ═══════════════════════════════════════════════════════════
#  主测试流程
# ═══════════════════════════════════════════════════════════

def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 80)
    log("  Memento 嵌入模型对比测试")
    log(f"  数据: {DATA_PATH.name} | 查询数: {len(TEST_QUERIES)}")
    log("=" * 80)

    # 加载数据
    memories = load_memories(DATA_PATH)
    log(f"\n  加载 {len(memories)} 条记忆")

    # 统计标签分布
    tag_counts = defaultdict(int)
    for m in memories:
        for t in m.get("tags", []):
            tag_counts[t] += 1
    log(f"  标签种类: {len(tag_counts)}")

    # 建 tag->ids 映射（用于命中率评估）
    tag_to_ids = defaultdict(set)
    for m in memories:
        for t in m.get("tags", []):
            tag_to_ids[t].add(m["id"])

    # ═══════════════════════════════════════════════
    #  逐后端测试
    # ═══════════════════════════════════════════════

    all_results = {}  # backend_name -> [query_stats]

    for backend_name, cfg in BACKENDS.items():
        log(f"\n{'━' * 35} {backend_name} {'━' * 35}")

        try:
            log(f"  模型: {cfg['model_name']}")
            vi, build_time = build_index_direct(
                cfg["model_name"], cfg["device"], memories)
            log(f"  索引构建: {build_time:.2f}s | 维度: {vi.dimension} | "
                f"节点数: {vi.size}")
        except Exception as e:
            log(f"  *** 加载失败: {e}")
            all_results[backend_name] = None
            continue

        backend_results = []

        for query, target_tag in TEST_QUERIES:
            stats = run_query(vi, query, k=10, target_tag=target_tag)
            results = stats["results"]

            # 计算标签命中率
            if target_tag and target_tag in tag_to_ids:
                target_ids = tag_to_ids[target_tag]
                hits = sum(1 for nid, _ in results if nid in target_ids)
                stats["target_hits"] = hits
                stats["target_total"] = len(target_ids)
            else:
                stats["target_hits"] = None

            backend_results.append(stats)

            # 打印
            hit_str = ""
            if stats["target_hits"] is not None:
                hit_str = f" | 命中: {stats['target_hits']}/{stats['target_total']}"

            log(f"\n  Q: {query}")
            log(f"    分数: top={stats['top_score']:.4f}  bot={stats['bot_score']:.4f}  "
                f"spread={stats['score_spread']:.4f}  std={stats['score_std']:.4f}"
                f"{hit_str}")
            log(f"    耗时: encode={stats['encode_time_ms']:.1f}ms  "
                f"search={stats['search_time_ms']:.2f}ms")

            for i, (nid, score) in enumerate(results[:5], 1):
                # 查找对应文本
                text = ""
                for m in memories:
                    if m["id"] == nid:
                        text = m["text"][:45]
                        break
                log(f"    {i}. [{score:.4f}] {text}")

        all_results[backend_name] = backend_results

        # 清理 GPU 内存
        try:
            del vi
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ═══════════════════════════════════════════════
    #  汇总对比
    # ═══════════════════════════════════════════════

    log(f"\n\n{'=' * 80}")
    log("  汇总对比")
    log("=" * 80)

    # 分数分布汇总
    log(f"\n  {'后端':<15} {'平均 spread':<14} {'平均 std':<12} {'平均 top':<12} {'编码耗时(ms)':<16}")
    log(f"  {'-'*65}")

    for backend_name, results_list in all_results.items():
        if results_list is None:
            log(f"  {backend_name:<15} {'FAILED':<14}")
            continue

        spreads = [r["score_spread"] for r in results_list]
        stds = [r["score_std"] for r in results_list]
        tops = [r["top_score"] for r in results_list]
        enc_times = [r["encode_time_ms"] for r in results_list]

        log(f"  {backend_name:<15} {np.mean(spreads):<14.4f} "
            f"{np.mean(stds):<12.4f} {np.mean(tops):<12.4f} "
            f"{np.mean(enc_times):<16.1f}")

    # 命中率汇总
    log(f"\n  标签命中率（有目标标签的查询）:")
    log(f"  {'后端':<15} {'总命中/总数':<15} {'命中率':<10}")
    log(f"  {'-'*40}")

    for backend_name, results_list in all_results.items():
        if results_list is None:
            continue

        total_hits = sum(r["target_hits"] for r in results_list
                         if r["target_hits"] is not None)
        total_targets = sum(r.get("target_total", 0) for r in results_list
                            if r["target_hits"] is not None)

        if total_targets > 0:
            rate = total_hits / total_targets
            log(f"  {backend_name:<15} {total_hits}/{total_targets * 10:<12} "
                f"{rate*100:.1f}%")

    # 保存结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    log(f"\n  结果已保存到: {OUTPUT_PATH}")
    log("\n  测试完成!")


if __name__ == "__main__":
    main()
