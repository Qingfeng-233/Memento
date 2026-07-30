"""
诊断：4B embedding 下，概念图扩散到底有没有把候选推进 top-k。

输出：
  1. 每个查询的 query_with_concepts top-k，带 rag_score / concept_score 分量
  2. 同一查询的纯 RAG top-k（query_rag_only）
  3. 对比：概念融合后有多少条是 RAG 没召回但概念拉进来的（"概念增益"）
  4. concept_score 的实际数值范围——如果全是 0，说明概念图完全没生效
"""
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

QUERIES = [
    "容器启动时配置丢失怎么排查",
    "钢琴连电脑需要什么线和软件",
    "为什么流行歌都是情情爱爱",
    "手机传文件到电脑用什么软件",
    "独立游戏开发者要不要学美术",
    "梯子和局域网冲突怎么解决",
]


def parse(path, limit=None):
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


def truncate(t, n=70):
    t = re.sub(r"\s+", " ", t).strip()
    return t[:n] + "…" if len(t) > n else t


def main():
    from memento.api import Memento

    pairs = parse(ROOT / "data" / "testtxt.txt")
    print(f"data: {len(pairs)} pairs")

    engine = Memento(embedding_model="api:Qwen/Qwen3-Embedding-4B", diffusion_hops=2)
    for i, p in enumerate(pairs):
        engine.add_node(f"用户问: {p['q']}\n回答: {p['a']}",
                        node_id=f"qa_{i:04d}", importance=0.5)
    engine.build_index()
    engine.build_concept_graph(
        top_k=8,
        keyword_model="models/Qwen3-Embedding-0.6B",
        keyword_device=None, keyword_dtype="float16",
        keyword_cache_enabled=True, keyword_cache_dir="data/keyatten_cache",
        keyword_sim_threshold=0.45, keyword_temperature=0.06,
        keyword_top_neighbors=8, max_concepts=500, min_concept_energy=0.5,
        use_surprisal=True,
    )

    print(f"\n{'='*90}")
    print("  诊断：概念图扩散对 top-k 的实际贡献（4B embedding）")
    print(f"{'='*90}")

    total_concept_boosted = 0
    total_queries = 0

    for q in QUERIES:
        total_queries += 1
        # 概念融合检索（带分量）
        fused = engine.query_with_concepts(
            q, k=5, seed_k=20, concept_k=10, concept_hops=2, concept_weight=0.45)
        # 纯 RAG
        rag_only = engine.query_rag_only(q, k=5)

        fused_ids = [h["id"] for h in fused]
        rag_ids = [h["id"] for h in rag_only]

        # 概念增益：融合 top-k 里有，但纯 RAG top-k 里没有的
        only_in_fused = set(fused_ids) - set(rag_ids)
        only_in_rag = set(rag_ids) - set(fused_ids)

        print(f"\n{'─'*90}")
        print(f"  查询：{q}")
        print(f"{'─'*90}")
        print(f"  纯RAG top5 id: {rag_ids}")
        print(f"  融合 top5 id:  {fused_ids}")
        print(f"  概念拉入(融合有/RAG无): {len(only_in_fused)} 条")
        print(f"  RAG 独有(被挤出): {len(only_in_rag)} 条")

        # 看 concept_score 分布
        cscores = [h.get("concept_score", 0) for h in fused]
        rscores = [h.get("rag_score", 0) for h in fused]
        max_c = max(cscores) if cscores else 0
        max_r = max(rscores) if rscores else 0
        print(f"  分量范围: rag=[{min(rscores):.3f}~{max_r:.3f}]  "
              f"concept=[{min(cscores):.3f}~{max_c:.3f}]")

        # 逐条展示
        print(f"\n  {'#':<3} {'id':<12} {'final':>7} {'rag':>7} {'concept':>7} "
              f"{'0.45*c':>7}  {'文本':<40}")
        for i, h in enumerate(fused):
            r = h.get("rag_score", 0)
            c = h.get("concept_score", 0)
            f = h.get("score", 0)
            boosted = "◆概念拉入" if h["id"] in only_in_fused else ""
            print(f"  {i+1:<3} {h['id']:<12} {f:>7.3f} {r:>7.3f} {c:>7.3f} "
                  f"{0.45*c:>7.3f}  {truncate(h['text'],40):<40} {boosted}")

        if only_in_fused:
            total_concept_boosted += len(only_in_fused)

    print(f"\n{'='*90}")
    print(f"  汇总：{total_queries} 个查询，概念扩散共拉入 {total_concept_boosted} 条候选")
    if total_concept_boosted == 0:
        print("  ⚠ 概念扩散对 top-k 零贡献——退化成纯 RAG")
    else:
        print(f"  概念扩散有贡献（{total_concept_boosted} 条增益）")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
