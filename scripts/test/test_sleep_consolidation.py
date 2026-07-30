"""
阶段 2 验收：跑一次带 LLM 裁决建边 + 节点融合的睡眠周期。

不跑量化指标。dump 出：
  - 睡眠报告（含 LLM 调用数 / 缓存命中 / 融合数）
  - 每个融合节点的源节点对 + LLM 合成文本（人眼判断是否合理近重复）
  - LLM 建边前后的边数对比

判断标准：融合案例是不是真重复（不是误融合）；LLM 边是不是合理语义关联。
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


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


def truncate(text: str, limit: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "…" if len(text) > limit else text


def main() -> None:
    import argparse
    from memento.api import Memento

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40,
                        help="只用前 N 对做验收（默认 40，全量 145 太慢）")
    parser.add_argument("--llm-budget", type=int, default=15)
    parser.add_argument("--fusion-max", type=int, default=8)
    args = parser.parse_args()

    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt", limit=args.limit)
    print(f"data: {len(pairs)} Q&A pairs (limit={args.limit})\n")

    model = str(ROOT / "models" / "Qwen3-Embedding-0.6B")
    engine = Memento(
        embedding_model=model,
        diffusion_hops=2,
        sleep_llm_curate=True,
        sleep_fusion=True,
        fusion_cosine_threshold=0.85,
        llm_curate_cosine=0.35,
        llm_max_calls_per_cycle=args.llm_budget,
    )
    engine.sleep_engine.fusion_max_per_cycle = args.fusion_max

    for i, pair in enumerate(pairs):
        engine.add_node(memory_text(pair), node_id=f"qa_{i:04d}", importance=0.5)
    engine.build_index()

    edges_before = engine.graph.edge_count

    print("触发睡眠周期（含 LLM 裁决 + 融合）...\n")
    t0 = time.time()
    report = engine.trigger_sleep()
    elapsed = time.time() - t0

    edges_after = engine.graph.edge_count

    print(report.summary())
    print(f"  睡眠耗时: {elapsed:.1f}s\n")

    # ── 融合案例 ──────────────────────────────────────────
    fusion_nodes = [
        n for n in engine.graph.nodes.values()
        if "__fusion__" in n.tags
    ]
    print("=" * 70)
    print(f"  融合案例：{len(fusion_nodes)} 个")
    print("=" * 70)
    for fn in fusion_nodes:
        print(f"\n  ◆ {fn.id}  (fused_from {len(fn.fused_from)} 源)")
        print(f"    合成文本: {truncate(fn.text, 100)}")
        for src_id in fn.fused_from:
            src = engine.graph.get_node(src_id)
            if src:
                print(f"    ├ 源 {src_id} [{src.status}, superseded_by={src.superseded_by}]")
                print(f"    │   {truncate(src.text, 90)}")

    # ── LLM 边抽查（前 15 条）─────────────────────────────
    llm_edges = [
        (s, t, e) for s, t, e in engine.graph.get_all_edges()
        if e.edge_type == "llm"
    ]
    print("\n" + "=" * 70)
    print(f"  LLM 裁决边：{len(llm_edges)} 条（前 15 条）")
    print("=" * 70)
    for s, t, e in llm_edges[:15]:
        na = engine.graph.get_node(s)
        nb = engine.graph.get_node(t)
        print(f"\n  [{e.weight:.3f}] {s} <-> {t}")
        if na:
            print(f"    A: {truncate(na.text, 80)}")
        if nb:
            print(f"    B: {truncate(nb.text, 80)}")
    if len(llm_edges) > 15:
        print(f"\n  ... 还有 {len(llm_edges) - 15} 条")

    # ── 统计 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  边统计")
    print("=" * 70)
    by_type: dict[str, int] = {}
    for s, t, e in engine.graph.get_all_edges():
        by_type[e.edge_type] = by_type.get(e.edge_type, 0) + 1
    for et, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {et:<15} {cnt}")
    print(f"    {'合计':<15} {edges_after}  (睡眠前 {edges_before}, Δ{edges_after-edges_before:+d})")

    # 被融合的源节点状态确认
    superseded = [n for n in engine.graph.nodes.values() if n.status == "superseded"]
    print(f"\n  superseded 节点数: {len(superseded)} (应 = 融合数 × 2 = {len(fusion_nodes)*2})")


if __name__ == "__main__":
    main()
