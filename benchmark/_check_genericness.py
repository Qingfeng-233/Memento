"""查 USB / LocalSend / 开心 的 genericness 实际值。"""
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

    # 手动走 build_concept_graph 的逻辑，截获 genericness
    # 先抽关键词
    from memento.index.keyatten_extractor import MemoryKeywordExtractor
    device = "cuda"
    ext = MemoryKeywordExtractor(model_path="models/Qwen3-Embedding-0.6B",
                                  device=device, dtype="float16", default_top_k=8)
    all_nodes = sorted(engine.graph.nodes.items(), key=lambda x: x[0])
    node_ids = [nid for nid, _ in all_nodes]
    texts = [n.text for _, n in all_nodes]
    total_docs = len(texts)

    engine._node_keywords = {}
    doc_freq = {}
    ext.update_idf(texts)
    for nid, text in zip(node_ids, texts):
        kws = ext.extract(text, top_k=8)
        engine._node_keywords[nid] = kws
        for kw in set(kws):
            doc_freq[kw] = doc_freq.get(kw, 0) + 1

    candidate_keywords = sorted(kw for kw, freq in doc_freq.items() if 0 < freq <= 30)
    cand_vectors = engine.vector_index.encode(candidate_keywords, mode="document")

    sims = cosine_matrix(np.vstack(cand_vectors))
    np.fill_diagonal(sims, 0.0)
    avg_sims = sims.mean(axis=1)
    mn, mx = float(avg_sims.min()), float(avg_sims.max())
    print(f"avg_sims range: [{mn:.4f}, {mx:.4f}]")
    print(f"(范围太窄会导致 genericness 区分度差)\n")

    kw_to_gen = {}
    for i, kw in enumerate(candidate_keywords):
        raw = float(avg_sims[i])
        gen = (raw - mn) / (mx - mn) if mx > mn else 0.0
        kw_to_gen[kw] = gen

    # 看重点词
    targets = ["USB", "LocalSend", "Syncthing", "v2rayN",
               "MIDI to USB", "发现钢琴", "电脑扫描手机",
               "难过", "羁绊", "软件"]
    print(f"{'词':<16} {'avg_sim':>8} {'generic':>8}")
    print("-"*40)
    for t in targets:
        if t in kw_to_gen:
            idx = candidate_keywords.index(t)
            print(f"{t:<16} {avg_sims[idx]:>8.4f} {kw_to_gen[t]:>8.4f}")
        else:
            print(f"{t:<16} 不在候选词里")

    # 看 genericness 分布
    gens = sorted(kw_to_gen.values())
    print(f"\ngenericness 分布:")
    print(f"  min={gens[0]:.4f}  25%={gens[len(gens)//4]:.4f}  "
          f"50%={gens[len(gens)//2]:.4f}  75%={gens[3*len(gens)//4]:.4f}  max={gens[-1]:.4f}")

    # USB 在分位数里的位置
    usb_gen = kw_to_gen.get("USB", -1)
    rank = sum(1 for g in gens if g < usb_gen)
    print(f"\nUSB genericness={usb_gen:.4f}, 排名 {rank}/{len(gens)} (越高越通用)")


if __name__ == "__main__":
    main()
