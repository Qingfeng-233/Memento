"""查 file/USB/钢琴/LocalSend 在各自节点里的惊奇度，看能不能救回误杀的词。"""
import re, sys, math
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
    from memento.concept.concept_graph import ConceptGraph

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

    # 算惊奇度：每个节点的每个关键词
    all_kws = list(set(kw for kws in engine._node_keywords.values() for kw in kws))
    kw_vecs = engine.vector_index.encode(all_kws, mode="document")
    kw_to_vec = dict(zip(all_kws, kw_vecs))

    print(f"{'关键词':<14} {'出现的节点':<10} {'cos(词,文本)':>12} {'惊奇度':>7}")
    print("-" * 55)

    targets = ["file", "USB", "MIDI", "钢琴", "软件", "电脑",
               "LocalSend", "Syncthing", "容器配置", "开心", "难过"]

    for t in targets:
        # 找出这个关键词出现在哪些节点
        for nid, kws in engine._node_keywords.items():
            if t in kws:
                node = engine.graph.get_node(nid)
                if node and node.vector is not None and t in kw_to_vec:
                    cos = float(np.dot(node.vector, kw_to_vec[t]))
                    surprisal = 1.0 - cos
                    snippet = re.sub(r"\s+", " ", node.text).strip()[:40]
                    print(f"{t:<14} {nid:<10} {cos:>12.4f} {surprisal:>7.4f}  {snippet}")

    # 重点：对比 file 和 LocalSend 的惊奇度
    print(f"\n{'='*60}")
    print("核心对比：file（垃圾词）vs LocalSend（专有词）的惊奇度")
    print(f"{'='*60}")
    for t in ["file", "USB", "LocalSend", "MIDI", "钢琴"]:
        surps = []
        for nid, kws in engine._node_keywords.items():
            if t in kws:
                node = engine.graph.get_node(nid)
                if node and node.vector is not None and t in kw_to_vec:
                    cos = float(np.dot(node.vector, kw_to_vec[t]))
                    surps.append(1.0 - cos)
        if surps:
            print(f"  {t:<14} 惊奇度: {surps}  平均={sum(surps)/len(surps):.4f}")


if __name__ == "__main__":
    main()
