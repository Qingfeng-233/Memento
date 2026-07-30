"""查 file/USB/软件/电脑/手机/线 这些泛词的 df、genericness、energy、是否情感豁免。"""
import re, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import numpy as np
from memento.concept.concept_graph import ConceptGraph, cosine_matrix


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
    )
    cg = engine.concept_graph
    total_docs = len(pairs)

    # 泛词列表
    generic = ["file", "USB", "软件", "电脑", "手机", "线", "MIDI", "Korg D1"]
    print(f"{'词':<12} {'在图?':>5} {'df':>3} {'energy':>7} {'连了几个event':>12}  {'event列表'}")
    print("-" * 80)
    for w in generic:
        cid = cg.concept_id(w)
        c = cg.concepts.get(cid)
        if c is None:
            # 看看它在不在 doc_freq 里（被 energy 过滤了）
            df = 0
            for nid, kws in engine._node_keywords.items():
                if w in kws:
                    df += 1
            print(f"{w:<12} {'否':>5} {'—':>3} {'—':>7} {'—':>12}  (df={df}, 被过滤)")
            continue
        events = cg.concept_to_events.get(cid, {})
        ev_list = list(events.keys())
        print(f"{w:<12} {'是':>5} {c.doc_freq:>3} {c.initial_energy:>7.4f} {len(events):>12}  {ev_list}")

    # 看 file 到底出现在哪些节点的关键词里
    print(f"\n=== 'file' 出现在哪些节点的关键词里 ===")
    for nid, kws in engine._node_keywords.items():
        if "file" in kws:
            node = engine.graph.get_node(nid)
            t = re.sub(r"\s+", " ", node.text).strip()[:60] if node else "?"
            print(f"  {nid}: {t}")

    # max_node_freq 是多少
    print(f"\nmax_node_freq 默认值 = 30（df>30 的词会被过滤）")
    print(f"min_concept_energy = 0.5")


if __name__ == "__main__":
    main()
