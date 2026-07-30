"""
关键词语义去重 A/B 对照。

不跑量化指标（Hit@k/MRR 容易"跑分高效果差"）。改为把开关两边的
top-k 结果并排 dump 出来 + 概念图诊断，靠读结果判断"偏没偏、信息够不够"。

对照维度：
  A. dedup_concepts=False （现状：精确串各自独立概念）
  B. dedup_concepts=True  （余弦贪心合并近义锚点）

诊断重点：
  - "服务配置" / "环境配置缺失" 这类近义锚点被怎么合并
  - 概念数 / 事件-概念边 / 概念-概念边的变化
  - 每个查询 top-k 文本并排，看排序有没有跑偏
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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

# 建图参数（对齐 benchmark/compare_memory_systems.py 的 MementoAdapter.build）
CONCEPT_KWARGS = dict(
    top_k=8,
    keyword_model="models/Qwen3-Embedding-0.6B",
    keyword_device=None,
    keyword_dtype="float16",
    keyword_cache_enabled=True,
    keyword_cache_dir="data/keyatten_cache",
    keyword_sim_threshold=0.45,
    keyword_temperature=0.06,
    keyword_top_neighbors=8,
    max_concepts=500,
    min_concept_energy=0.5,
)
QUERY_KWARGS = dict(
    k=5,
    seed_k=20,
    concept_k=10,
    concept_hops=2,
    concept_weight=0.45,
)


def parse_chat_data(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    content = path.read_text(encoding="utf-8-sig")
    pairs: list[dict[str, str]] = []
    for part in re.split(r"【用户提问】", content):
        part = part.strip()
        if not part or "【AI 回答】" not in part:
            continue
        question, answer = part.split("【AI 回答】", 1)
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def memory_text(pair: dict[str, str]) -> str:
    return f"用户问: {pair['question']}\n回答: {pair['answer']}"


def truncate(text: str, limit: int = 70) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "…" if len(text) > limit else text


def build_engine(pairs: list[dict[str, str]], dedup: bool,
                 threshold: float = 0.82) -> tuple[object, dict]:
    from memento.api import Memento

    model = str(ROOT / "models" / "Qwen3-Embedding-0.6B")
    engine = Memento(embedding_model=model, diffusion_hops=2)
    for index, pair in enumerate(pairs):
        engine.add_node(
            text=memory_text(pair),
            node_id=f"qa_{index:04d}",
            importance=0.5,
        )
    engine.build_index()
    t0 = time.time()
    info = engine.build_concept_graph(
        dedup_concepts=dedup, dedup_threshold=threshold, **CONCEPT_KWARGS
    )
    info["build_seconds"] = time.time() - t0
    return engine, info


def query_topk(engine, query: str) -> list[dict]:
    return engine.query_with_concepts(query, **QUERY_KWARGS)


def find_merges_touching(merge_log: list[dict], needles: list[str]) -> list[dict]:
    hits = []
    for m in merge_log:
        if any(n in m["from"] or n in m["to"] for n in needles):
            hits.append(m)
    return hits


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.82,
                        help="去重余弦阈值（0.82=松，0.90=严）")
    args = parser.parse_args()
    threshold = args.threshold

    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"data: {len(pairs)} Q&A pairs")
    print(f"dedup threshold: {threshold}\n")

    print("构建 A: dedup_concepts=False ...")
    engine_a, info_a = build_engine(pairs, dedup=False)
    print(f"构建 B: dedup_concepts=True (threshold={threshold}) ...")
    engine_b, info_b = build_engine(pairs, dedup=True, threshold=threshold)

    # ── 概念图诊断 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  概念图诊断")
    print("=" * 70)
    print(f"  {'指标':<20} {'A (不去重)':<15} {'B (去重)':<15} {'差值':<10}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")
    for key in ("concepts", "event_concept_edges", "concept_edges", "vocab_size"):
        a, b = info_a.get(key, 0), info_b.get(key, 0)
        print(f"  {key:<20} {a:<15} {b:<15} {b-a:+d}")

    print(f"\n  B 合并掉的关键词对数: {info_b.get('dedup_merge_count', 0)}")
    merges = info_b.get("dedup_merges", [])
    if merges:
        # 重点看 服务配置 / 环境配置缺失 这类
        highlights = find_merges_touching(merges, ["服务配置", "美少", "少女"])
        if highlights:
            print("\n  >>> 重点案例（含 '服务配置' 类锚点）:")
            for m in highlights:
                print(f"      {m['from']:<14} -> {m['to']}")
        # 其余合并对，限制输出量
        rest = [m for m in merges if m not in highlights]
        if rest:
            print(f"\n  其余合并对（前 30 条，共 {len(rest)} 条）:")
            for m in rest[:30]:
                print(f"      {m['from']:<14} -> {m['to']}")
            if len(rest) > 30:
                print(f"      ... 还有 {len(rest) - 30} 条")

    # ── 每查询 top-k 并排 ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  检索结果对照（A=不去重  B=去重）")
    print("=" * 70)

    for query in QUERIES:
        top_a = query_topk(engine_a, query)
        top_b = query_topk(engine_b, query)
        ids_a = {r["id"] for r in top_a}
        ids_b = {r["id"] for r in top_b}

        print(f"\n  ▸ {query}")
        print(f"    重叠 {len(ids_a & ids_b)}/{len(ids_a | ids_b)}  "
              f"| 仅A: {len(ids_a - ids_b)}  仅B: {len(ids_b - ids_a)}")

        max_rows = max(len(top_a), len(top_b))
        print(f"    {'#':<3} {'A (不去重)':<60} {'B (去重)':<60}")
        print(f"    {'-'*3} {'-'*60} {'-'*60}")
        for i in range(max_rows):
            left = top_a[i] if i < len(top_a) else None
            right = top_b[i] if i < len(top_b) else None
            left_txt = truncate(left["text"], 55) if left else ""
            right_txt = truncate(right["text"], 55) if right else ""
            left_mark = " " if left and left["id"] in ids_b else "●"  # ● = 仅 A
            right_mark = " " if right and right["id"] in ids_a else "●"  # ● = 仅 B
            left_s = f"{left['score']:.3f}" if left else "     "
            right_s = f"{right['score']:.3f}" if right else "     "
            print(f"    {i+1:<3} {left_mark}{left_s} {left_txt:<52} "
                  f"{right_mark}{right_s} {right_txt:<52}")

    print("\n" + "=" * 70)
    print("  图例: ● 表示该结果只出现在这一侧（另一侧没有）")
    print("  判断指引: 看 ● 行的文本——是 B 引入了无关结果(跑偏)，")
    print("           还是 A 遗漏了相关结果(B 补回)。")
    print("=" * 70)


if __name__ == "__main__":
    main()
