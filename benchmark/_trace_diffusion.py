"""
追溯：手机传文件查询里，qa_0017（钢琴荒废）是怎么被概念图扩散拉进来的。

输出：
  1. 这个查询激活了哪些种子概念
  2. 扩散后哪些概念被激活
  3. qa_0017 连着哪些概念（concept_to_events 反查）
  4. 交叉：哪些概念既被查询激活、又连着 qa_0017 ——那就是扩散路径
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np


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
    )

    cg = engine.concept_graph
    query = "手机传文件到电脑用什么软件"
    target_id = "qa_0017"  # 钢琴荒废

    # 1. 种子概念：查询向量 vs 概念向量
    qvec = engine.vector_index.encode([query], mode="query")[0]
    concept_items = list(cg.concepts.items())
    concept_ids = [cid for cid, _ in concept_items]
    concept_vecs = np.vstack([c.vector for _, c in concept_items])
    q = qvec.astype(np.float32).reshape(1, -1)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn
    sims = (concept_vecs @ q.T).reshape(-1)

    print(f"查询：{query}")
    print(f"目标：{target_id}（钢琴荒废）\n")

    # 种子概念 top-10
    seed_order = np.argsort(-sims)[:10]
    print("=== 种子概念（查询激活的 top-10）===")
    seed_concepts = {}
    for idx in seed_order:
        cid = concept_ids[int(idx)]
        c = cg.concepts[cid]
        sim = float(sims[int(idx)])
        act = max(0.0, sim) * c.initial_energy
        seed_concepts[cid] = act
        print(f"  {c.text:<20} sim={sim:.3f} energy={c.initial_energy:.3f} act={act:.3f}")

    # 2. 扩散
    activations = cg.diffuse(seed_concepts, hops=2)
    print(f"\n=== 扩散后激活的概念（top-20）===")
    for cid, act in sorted(activations.items(), key=lambda x: -x[1])[:20]:
        c = cg.concepts.get(cid)
        if c:
            print(f"  {c.text:<20} activation={act:.4f}")

    # 3. qa_0017 连着哪些概念
    print(f"\n=== {target_id} 连接的概念 ===")
    target_concepts = cg.event_to_concepts.get(target_id, {})
    for cid, w in sorted(target_concepts.items(), key=lambda x: -x[1]):
        c = cg.concepts.get(cid)
        if c:
            act = activations.get(cid, 0)
            is_seed = "★种子" if cid in seed_concepts else ""
            print(f"  {c.text:<20} edge_weight={w:.3f}  concept_activation={act:.4f}  {is_seed}")

    # 4. 交叉：qa_0017 的概念里，哪些在扩散后被激活了
    print(f"\n=== 扩散路径分析 ===")
    path_concepts = []
    for cid in target_concepts:
        act = activations.get(cid, 0)
        if act > 0.01:
            c = cg.concepts.get(cid)
            if c:
                path_concepts.append((c.text, act, target_concepts[cid]))
    if path_concepts:
        path_concepts.sort(key=lambda x: -x[1])
        print(f"  {target_id} 通过以下概念被扩散激活：")
        for text, act, ew in path_concepts:
            print(f"    概念「{text}」  activation={act:.4f}  event边权={ew:.3f}")
    else:
        print(f"  {target_id} 的概念都没被扩散激活，可能是直接 RAG 命中")

    # 5. 这些"桥"概念连着哪些别的 event（看它是不是真的跨主题）
    print(f"\n=== 桥概念连接的所有 event（看是否跨主题）===")
    for text, act, ew in path_concepts[:3]:
        cid = cg.concept_id(text)
        events = cg.concept_to_events.get(cid, {})
        print(f"\n  概念「{text}」(activation={act:.4f}) 连接 {len(events)} 个 event:")
        for eid, w in sorted(events.items(), key=lambda x: -x[1])[:8]:
            node = engine.graph.get_node(eid)
            t = re.sub(r"\s+", " ", node.text).strip()[:50] if node else "?"
            print(f"    {eid} (w={w:.3f}) {t}")


if __name__ == "__main__":
    main()
