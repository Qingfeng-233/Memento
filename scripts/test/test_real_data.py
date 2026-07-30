"""
Memento 温度缩放对比测试 — 真实聊天数据版

数据源: data/testtxt.txt (143 条真实 Q&A 对话)
对比: Qwen3-0.6B (τ=0) vs Qwen3-0.6B (τ=0.05)
评估: 纯 RAG 分数分布 + 扩散联想结果

优化: 只加载模型一次，构建索引一次，切换温度参数跑两组查询
"""

import sys
import os
import time
import re
import numpy as np
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memento.api import Memento


# ─────────────────────────────────────────────────────────────
#  数据解析
# ─────────────────────────────────────────────────────────────

def parse_chat_data(filepath: str) -> list:
    """解析 testtxt.txt，返回 [{question, answer, combined}, ...]"""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    parts = re.split(r'【用户提问】', content)
    pairs = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '【AI 回答】' in part:
            q, a = part.split('【AI 回答】', 1)
            q = q.strip()
            a = a.strip()
            if q:
                combined = f"用户问: {q} 回答: {a[:200]}"
                pairs.append({
                    "question": q,
                    "answer": a,
                    "combined": combined,
                })

    return pairs


# ─────────────────────────────────────────────────────────────
#  统计工具
# ─────────────────────────────────────────────────────────────

def score_stats(scores: list) -> dict:
    if not scores:
        return {}
    arr = np.array(scores)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "spread": float(arr.max() - arr.min()),
        "std": float(arr.std()),
        "mean": float(arr.mean()),
    }


def truncate(text: str, n: int = 70) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


# ─────────────────────────────────────────────────────────────
#  查询列表（和 V2 对比测试相同）
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────────────

def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")

    # ══════════════════════════════════════════════════════════
    #  1. 解析数据
    # ══════════════════════════════════════════════════════════
    print(f"解析数据: {data_path}")
    pairs = parse_chat_data(data_path)
    print(f"解析完成: {len(pairs)} 条记忆\n")

    # ══════════════════════════════════════════════════════════
    #  2. 构建系统（只加载模型一次）
    # ══════════════════════════════════════════════════════════
    print(f"{'━' * 72}")
    print(f"  构建 Memento: Qwen3-0.6B + 真实对话数据")
    print(f"{'━' * 72}")
    t0 = time.time()

    memento = Memento(
        embedding_model=model_path,
        device="cuda",
        diffusion_hops=2,
        diffusion_alpha=0.3,
        diffusion_beta=0.6,
        score_temperature=0.0,  # 先设 τ=0
    )

    # 添加节点
    for i, p in enumerate(pairs):
        memento.add_node(p["combined"], node_id=f"mem_{i:04d}")

    # 构建索引
    memento.build_index()

    # 情境共现建边（每 3 条对话为一个上下文窗口）
    node_ids = [f"mem_{i:04d}" for i in range(len(pairs))]
    for i in range(0, len(node_ids) - 2, 3):
        window = node_ids[i:i + 3]
        memento.activate(window)

    build_time = time.time() - t0
    print(f"  完成! 节点={memento.stats['total_nodes']}, "
          f"边={memento.stats['total_edges']}, 耗时={build_time:.1f}s\n")

    # ══════════════════════════════════════════════════════════
    #  3. 跑查询对比
    # ══════════════════════════════════════════════════════════
    K = 10
    SEED_K = 20
    temperatures = [0.0, 0.05]

    # 存储所有结果
    all_results = {}  # {tau: {query: {rag: [...], diff: [...]}}}

    for tau in temperatures:
        # 切换温度
        memento.vector_index._score_temperature = tau
        tau_label = f"τ={tau}" if tau > 0 else "τ=0(无缩放)"
        print(f"\n{'=' * 72}")
        print(f"  温度 {tau_label}")
        print(f"{'=' * 72}")

        all_results[tau] = {}

        for qi, query in enumerate(QUERIES, 1):
            # 纯 RAG
            rag = memento.query_rag_only(query, k=K)
            # 扩散联想
            diff = memento.query(query, k=K, seed_k=SEED_K)

            all_results[tau][query] = {"rag": rag, "diff": diff}

            rag_scores = [it["score"] for it in rag]
            diff_scores = [it["score"] for it in diff]
            rag_st = score_stats(rag_scores)
            diff_st = score_stats(diff_scores)

            print(f"\n  [{qi}/{len(QUERIES)}] 「{query}」")
            print(f"    RAG:  spread={rag_st['spread']:.4f}  std={rag_st['std']:.4f}"
                  f"  [{rag_st['min']:.3f} ~ {rag_st['max']:.3f}]")
            print(f"    扩散: spread={diff_st['spread']:.4f}  std={diff_st['std']:.4f}"
                  f"  [{diff_st['min']:.3f} ~ {diff_st['max']:.3f}]")

            # 打印 Top-5
            print(f"    ┌─ RAG Top-5 ─────────────────────────────────────")
            for i, it in enumerate(rag[:5], 1):
                print(f"    │ {i}. (sim={it['score']:.4f}) {truncate(it['text'])}")
            print(f"    └─────────────────────────────────────────────────")

            print(f"    ┌─ 扩散 Top-5 ────────────────────────────────────")
            for i, it in enumerate(diff[:5], 1):
                print(f"    │ {i}. (s={it['score']:.4f}) {truncate(it['text'])}")
            print(f"    └─────────────────────────────────────────────────")

    # ══════════════════════════════════════════════════════════
    #  4. 汇总对比
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 72}")
    print(f"  汇总: τ=0 vs τ=0.05 分数分布对比")
    print(f"{'=' * 72}")

    # 逐查询对比表
    print(f"\n  {'查询':<30s} | {'τ=0 RAG spread':>14s} | {'τ=0.05 RAG spread':>17s} | {'τ=0 扩散 spread':>15s} | {'τ=0.05 扩散 spread':>18s}")
    print(f"  {'-' * 30}-+-{'-' * 14}-+-{'-' * 17}-+-{'-' * 15}-+-{'-' * 18}")

    rag_spreads_0 = []
    rag_spreads_1 = []
    diff_spreads_0 = []
    diff_spreads_1 = []

    for query in QUERIES:
        r0 = score_stats([it["score"] for it in all_results[0.0][query]["rag"]])
        r1 = score_stats([it["score"] for it in all_results[0.05][query]["rag"]])
        d0 = score_stats([it["score"] for it in all_results[0.0][query]["diff"]])
        d1 = score_stats([it["score"] for it in all_results[0.05][query]["diff"]])

        rag_spreads_0.append(r0["spread"])
        rag_spreads_1.append(r1["spread"])
        diff_spreads_0.append(d0["spread"])
        diff_spreads_1.append(d1["spread"])

        q_short = query[:28]
        print(f"  {q_short:<30s} | {r0['spread']:>14.4f} | {r1['spread']:>17.4f} | {d0['spread']:>15.4f} | {d1['spread']:>18.4f}")

    # 平均值
    print(f"  {'-' * 30}-+-{'-' * 14}-+-{'-' * 17}-+-{'-' * 15}-+-{'-' * 18}")
    print(f"  {'平均':<30s} | {np.mean(rag_spreads_0):>14.4f} | {np.mean(rag_spreads_1):>17.4f}"
          f" | {np.mean(diff_spreads_0):>15.4f} | {np.mean(diff_spreads_1):>18.4f}")

    # 排名保持率
    print(f"\n  排名一致性分析 (Top-5 重叠):")
    for query in QUERIES:
        rag_0_ids = [it["id"] for it in all_results[0.0][query]["rag"][:5]]
        rag_1_ids = [it["id"] for it in all_results[0.05][query]["rag"][:5]]
        diff_0_ids = [it["id"] for it in all_results[0.0][query]["diff"][:5]]
        diff_1_ids = [it["id"] for it in all_results[0.05][query]["diff"][:5]]

        rag_overlap = len(set(rag_0_ids) & set(rag_1_ids))
        diff_overlap = len(set(diff_0_ids) & set(diff_1_ids))

        q_short = query[:25]
        print(f"    {q_short:<25s}  RAG重叠={rag_overlap}/5  扩散重叠={diff_overlap}/5")

    # 全局汇总
    all_rag_0 = []
    all_rag_1 = []
    all_diff_0 = []
    all_diff_1 = []
    for query in QUERIES:
        all_rag_0.extend([it["score"] for it in all_results[0.0][query]["rag"]])
        all_rag_1.extend([it["score"] for it in all_results[0.05][query]["rag"]])
        all_diff_0.extend([it["score"] for it in all_results[0.0][query]["diff"]])
        all_diff_1.extend([it["score"] for it in all_results[0.05][query]["diff"]])

    print(f"\n  全局分数分布 (跨所有查询):")
    for label, scores in [
        ("τ=0    RAG", all_rag_0),
        ("τ=0.05 RAG", all_rag_1),
        ("τ=0    扩散", all_diff_0),
        ("τ=0.05 扩散", all_diff_1),
    ]:
        st = score_stats(scores)
        print(f"    {label:<18s}: spread={st['spread']:.4f}  std={st['std']:.4f}"
              f"  range=[{st['min']:.3f}, {st['max']:.3f}]")

    print(f"\n{'=' * 72}")
    print(f"  测试完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
