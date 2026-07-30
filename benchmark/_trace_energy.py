"""溯源：USB / 引入软件 / 引入软件确实 为什么 initial_energy 那么高。"""
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


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
    info = engine.build_concept_graph(
        top_k=8, keyword_model="models/Qwen3-Embedding-0.6B",
        keyword_device=None, keyword_dtype="float16",
        keyword_cache_enabled=True, keyword_cache_dir="data/keyatten_cache",
        keyword_sim_threshold=0.45, keyword_temperature=0.06,
        keyword_top_neighbors=8, max_concepts=500, min_concept_energy=0.5,
    )

    cg = engine.concept_graph
    total_docs = len(pairs)

    # 公式
    print(f"total_docs = {total_docs}")
    print(f"公式: energy = 0.25 + 0.75 × specificity × idf_norm")
    print(f"  specificity: len<=1→0.15, <=2→0.45, <=4→0.75, >=5→1.0")
    print(f"  idf = log((N+1)/(df+1)) + 1")
    print(f"  idf_norm = min(1.0, idf / (log(N+1)+1))")
    print(f"  min_concept_energy 过滤阈值 = 0.5")
    print()

    # 查目标概念
    targets = ["USB", "引入软件", "引入软件确实", "电脑扫描手机", "LocalSend", "Syncthing",
               "MIDI to USB", "发现钢琴", "钢琴型号"]
    max_idf_norm = math.log(total_docs + 1) + 1.0
    print(f"idf_norm 的分母 = log({total_docs}+1)+1 = {max_idf_norm:.4f}")
    print()

    print(f"{'概念':<16} {'len':>3} {'spec':>5} {'df':>3} {'idf':>6} {'idf_norm':>8} {'energy':>7} {'公式展开':>30}")
    print("-" * 90)
    for t in targets:
        cid = cg.concept_id(t)
        c = cg.concepts.get(cid)
        if c is None:
            print(f"{t:<16}  (不在概念图里)")
            continue
        ln = len(t.strip())
        spec = ConceptGraph.specificity(t)
        df = c.doc_freq
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        idf_norm = min(1.0, idf / max_idf_norm)
        energy_calc = 0.25 + 0.75 * spec * idf_norm
        print(f"{t:<16} {ln:>3} {spec:>5.2f} {df:>3} {idf:>6.3f} {idf_norm:>8.4f} "
              f"{energy_calc:>7.4f}  0.25+0.75×{spec:.2f}×{idf_norm:.4f}")

    # 看看这些词出现在哪些 node 的关键词里
    print(f"\n{'='*90}")
    print("这些歧义词是从哪些节点的关键词里抽出来的？")
    print(f"{'='*90}")
    for t in targets:
        cid = cg.concept_id(t)
        found_in = []
        for nid, kws in engine._node_keywords.items():
            if t in kws:
                node = engine.graph.get_node(nid)
                snippet = re.sub(r"\s+", " ", node.text).strip()[:60] if node else "?"
                found_in.append(f"{nid}: {snippet}")
        print(f"\n  「{t}」 出现在 {len(found_in)} 个节点的关键词里:")
        for f in found_in[:5]:
            print(f"    {f}")


if __name__ == "__main__":
    main()
