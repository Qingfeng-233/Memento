"""
详细 debug：qa_0017（钢琴荒废）在"手机传文件"查询里是怎么到 #4 的。

追踪每一层：
  1. RAG 直接命中？查询 vs qa_0017 的余弦
  2. 概念激活了哪些种子？这些种子哪些连着 qa_0017？
  3. 每条路径的 surprisal 和边权是多少？
  4. 最终 concept_score 是怎么累加的？
"""
import re, sys, math, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import numpy as np


def parse(path):
    content = path.read_text(encoding="utf-8-sig")
    pairs = []
    for part in re.split(r"【用户提问】", content):
        part = part.strip()
        if not part or "【AI 回答】" not in part:
            continue
        q, a = part.split("【AI 回答】", 1)
        if q.strip() and a.strip():
            pairs.append({"q": q.strip(), "a": a.strip()})
    return pairs


def main():
    from memento.api import Memento
    pairs = parse(ROOT / "data" / "testtxt.txt")
    engine = Memento(embedding_model="api:Qwen/Qwen3-Embedding-4B", diffusion_hops=2)
    for i, p in enumerate(pairs):
        engine.add_node(f"用户问: {p['q']}\n回答: {p['a']}",
                        node_id=f"qa_{i:04d}", importance=0.5)
    engine.build_index()
    engine.build_concept_graph(
        top_k=8, keyword_model="models/Qwen3-Embedding-0.6B",
        keyword_device=None, keyword_dtype="float16",
        keyword_cache_enabled=True, keyword_cache_dir="data/keyatten_cache",
        keyword_sim_threshold=0.45, keyword_temperature=0.06,
        keyword_top_neighbors=8, max_concepts=500, min_concept_energy=0.5,
        use_surprisal=True,
    )

    cg = engine.concept_graph
    query = "手机传文件到电脑用什么软件"
    target = "qa_0017"

    print(f"查询：{query}")
    print(f"目标：{target}（钢琴荒废）\n")

    # ── 层 1：RAG 直接命中？──
    print("=" * 70)
    print("层 1：RAG 直接命中")
    print("=" * 70)
    qa17 = engine.graph.get_node(target)
    qvec = engine.vector_index.encode([query], mode="query")[0]
    cos_rag = float(np.dot(qvec, qa17.vector))
    print(f"  cos(查询, qa_0017) = {cos_rag:.4f}")
    if cos_rag < 0.35:
        print(f"  → RAG 召回了它吗？看 rag_score（之前诊断是 {0.340}）")
    print(f"  qa_0017 rag_score（从之前诊断）= 0.340")
    print(f"  结论：RAG 勉强召回（分数低），但进了 seed_k=20 的候选池\n")

    # ── 层 2：概念激活了哪些种子？──
    print("=" * 70)
    print("层 2：查询激活了哪些种子概念")
    print("=" * 70)
    concept_items = list(cg.concepts.items())
    concept_ids = [cid for cid, _ in concept_items]
    concept_vecs = np.vstack([c.vector for _, c in concept_items])
    q = qvec.astype(np.float32).reshape(1, -1)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn
    sims = (concept_vecs @ q.T).reshape(-1)

    # 种子 top-10
    seed_concepts = {}
    print(f"  {'概念':<20} {'cos(查询)':>9} {'energy':>7} {'activation':>10}")
    print(f"  {'-'*20} {'-'*9} {'-'*7} {'-'*10}")
    for idx in np.argsort(-sims)[:10]:
        cid = concept_ids[int(idx)]
        c = cg.concepts[cid]
        sim = float(sims[int(idx)])
        act = max(0.0, sim) * c.initial_energy
        seed_concepts[cid] = act
        print(f"  {c.text:<20} {sim:>9.4f} {c.initial_energy:>7.4f} {act:>10.4f}")

    # ── 层 3：qa_0017 连着哪些概念？这些概念被激活了吗？──
    print(f"\n{'='*70}")
    print(f"层 3：qa_0017 连接的概念，哪些被种子激活了？")
    print(f"{'='*70}")
    target_concepts = cg.event_to_concepts.get(target, {})
    for cid, ew in sorted(target_concepts.items(), key=lambda x: -x[1]):
        c = cg.concepts.get(cid)
        if not c:
            continue
        is_seed = cid in seed_concepts
        act = seed_concepts.get(cid, 0)
        # 查 surprisal
        surp_cache = cg.event_to_concepts.get(target, {}).get(cid, 0)
        print(f"  概志「{c.text}」")
        print(f"    event边权={ew:.4f}  种子激活={act:.4f}  "
              f"energy={c.initial_energy:.4f}  {'★种子' if is_seed else ''}")

    # ── 层 4：扩散路径 ──
    print(f"\n{'='*70}")
    print(f"层 4：概念扩散——种子概念扩散后，哪些到达了连 qa_0017 的概念？")
    print(f"{'='*70}")
    activations = cg.diffuse(seed_concepts, hops=2)

    # 找 qa_0017 的所有概念，看扩散后的激活值
    contributing = []
    for cid in target_concepts:
        act = activations.get(cid, 0)
        ew = target_concepts[cid]
        c = cg.concepts.get(cid)
        if act > 0.01 and c:
            score = act * ew
            contributing.append((c.text, act, ew, score))
            print(f"  「{c.text}」 activation={act:.4f} × event边权={ew:.4f} = {score:.4f}")

    total_concept_score = sum(s for _, _, _, s in contributing)
    print(f"\n  qa_0017 的 concept_score（归一前）= {total_concept_score:.4f}")

    # ── 层 5：最终分数分解 ──
    print(f"\n{'='*70}")
    print(f"层 5：最终分数分解")
    print(f"{'='*70}")
    # 从之前诊断知道：rag=0.340, concept=0.314, final=0.482
    print(f"  rag_score = 0.340（RAG 直接命中，勉强进 seed_k）")
    print(f"  concept_score = 0.314（概念扩散贡献）")
    print(f"  final = rag + 0.45 × concept = 0.340 + 0.45 × 0.314 = {0.340 + 0.45*0.314:.3f}")
    print(f"  （之前诊断 final = 0.482）")

    print(f"\n  根因分析：")
    print(f"  - rag_score 0.340 不算高，但它进了 top5 因为别的候选更差")
    print(f"  - concept_score 0.314 来自扩散——哪个概念贡献的？看上面层 4")


if __name__ == "__main__":
    main()
