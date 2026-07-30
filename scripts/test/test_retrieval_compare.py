"""
检索质量对比：睡眠前 vs 睡眠后（含 LLM 裁决 + 融合）。

不跑量化指标。对每个查询，把基线 top-k 和融合后 top-k 并排 dump，
由人判断：
  - 融合源节点（status=superseded）是否正确从检索消失
  - 融合节点本身是否在相关查询里正确浮现
  - 相关性有没有变差 / 有没有引入无关结果

判断标准由人读文本，不优化数字。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# 选能覆盖不同融合主题 + 通用主题的查询
QUERIES = [
    # 融合主题（应该看到融合节点浮现、源节点消失）
    "古风音乐 明月天涯",                    # fusion_0001
    "系统一直觉型人格 思考快与慢",          # fusion_0002
    "MIDI 线还是音频接口 钢琴连电脑",       # fusion_0005
    "容器部署 配置排查",                    # fusion_0006
    # 通用主题（不应该被融合干扰）
    "怎么提高学习效率 晚上崩盘",
    "手机传文件到电脑",
    "梯子和局域网冲突",
]

QUERY_KWARGS = dict(k=6, seed_k=20, concept_k=10, concept_hops=2,
                    concept_weight=0.45)


def parse_chat_data(path: Path, limit: int | None = None):
    content = path.read_text(encoding="utf-8-sig")
    pairs = []
    for part in re.split(r"【用户提问】", content):
        part = part.strip()
        if not part or "【AI 回答】" not in part:
            continue
        q, a = part.split("【AI 回答】", 1)
        q, a = q.strip(), a.strip()
        if q and a:
            pairs.append({"q": q, "a": a})
        if limit and len(pairs) >= limit:
            break
    return pairs


def mtext(p):
    return f"用户问: {p['q']}\n回答: {p['a']}"


def truncate(text: str, limit: int = 75) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "…" if len(text) > limit else text


def build_engine(pairs, with_consolidation: bool):
    from memento.api import Memento
    model = str(ROOT / "models" / "Qwen3-Embedding-0.6B")
    engine = Memento(
        embedding_model=model,
        diffusion_hops=2,
        sleep_llm_curate=with_consolidation,
        sleep_fusion=with_consolidation,
        llm_max_calls_per_cycle=30,
    )
    engine.sleep_engine.fusion_max_per_cycle = 10
    for i, p in enumerate(pairs):
        engine.add_node(mtext(p), node_id=f"qa_{i:04d}", importance=0.5)
    engine.build_index()
    engine.build_concept_graph(
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
    if with_consolidation:
        print("  触发睡眠周期（含 LLM 裁决 + 融合）...")
        t0 = time.time()
        report = engine.trigger_sleep()
        print(f"  睡眠完成 {time.time()-t0:.0f}s: "
              f"融合 {report.fusions_created} 个 / LLM 边 {report.llm_edges_created} 条")
    return engine


def run_queries(engine, label):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    results = {}
    for q in QUERIES:
        hits = engine.query_with_concepts(q, **QUERY_KWARGS)
        results[q] = hits
    return results


def print_side_by_side(baseline, post, engine_post):
    """并排打印基线 vs 融合后的 top-k。"""
    # 收集所有融合节点 id 和它们的源 id，用于标注
    fusion_ids = {n.id: n.fused_from
                  for n in engine_post.graph.nodes.values()
                  if "__fusion__" in n.tags}
    superseded_ids = {n.id: n.superseded_by
                      for n in engine_post.graph.nodes.values()
                      if n.status == "superseded"}

    for q in QUERIES:
        a_hits = baseline.get(q, [])
        b_hits = post.get(q, [])
        print(f"\n  ▸ {q}")
        max_rows = max(len(a_hits), len(b_hits))
        for i in range(max_rows):
            left = a_hits[i] if i < len(a_hits) else None
            right = b_hits[i] if i < len(b_hits) else None
            # 标注：融合节点 ◆ / superseded 源 ✗
            def tag(hit, side):
                if hit is None:
                    return ""
                nid = hit["id"]
                if nid in fusion_ids:
                    return "◆融合"
                if nid in superseded_ids:
                    return "✗源"
                return ""
            ls = f"{left['score']:.3f}" if left else "     "
            rs = f"{right['score']:.3f}" if right else "     "
            lt = truncate(left["text"], 60) if left else ""
            rt = truncate(right["text"], 60) if right else ""
            lm = tag(left, "L")
            rm = tag(right, "R")
            print(f"    {i+1:<2} {ls} {lm:<5} {lt:<55} | {rs} {rm:<5} {rt:<55}")


def main():
    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"data: {len(pairs)} Q&A pairs")

    print("\n构建基线引擎（无睡眠）...")
    engine_a = build_engine(pairs, with_consolidation=False)
    baseline = run_queries(engine_a, "基线（无融合）")

    print("\n构建融合引擎（睡眠后）...")
    engine_b = build_engine(pairs, with_consolidation=True)
    post = run_queries(engine_b, "融合后")

    # 诊断：查询时 superseded 节点有没有漏回来
    print("\n" + "="*72)
    print("  诊断：superseded 节点是否出现在检索结果里（应为 0）")
    print("="*72)
    leak_count = 0
    superseded_ids = {n.id for n in engine_b.graph.nodes.values()
                      if n.status == "superseded"}
    for q, hits in post.items():
        for h in hits:
            if h["id"] in superseded_ids:
                leak_count += 1
                print(f"  ⚠ 漏回: {q} → {h['id']}")
    if leak_count == 0:
        print("  ✓ 无漏回（所有 superseded 源节点都从检索消失）")
    else:
        print(f"  ✗ 漏回 {leak_count} 次")

    # 并排对比
    print_side_by_side(baseline, post, engine_b)
    print("\n  图例: ◆融合=该位置是融合节点  ✗源=该位置是已 superseded 的源节点")
    print("        （✗源 不应出现在融合后结果里）")


if __name__ == "__main__":
    main()
