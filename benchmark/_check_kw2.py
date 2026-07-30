"""清缓存后重建，检查 qa_0017 关键词和引入软件是否还在。"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def parse(path):
    content = path.read_text(encoding="utf-8-sig")
    pairs = []
    for part in re.split("【用户提问】", content):
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

    print("qa_0017 关键词:", engine._node_keywords.get("qa_0017"))
    print()
    found = False
    for nid, kws in engine._node_keywords.items():
        if "引入软件" in kws or "引入软件确实" in kws:
            print(f"  {nid}: {kws}")
            found = True
    if not found:
        print("引入软件/引入软件确实 不在任何节点的关键词里！")
    print()
    print("概念图里有 引入软件 吗:",
          "kw:引入软件" in engine.concept_graph.concepts)
    print("概念图里有 引入软件确实 吗:",
          "kw:引入软件确实" in engine.concept_graph.concepts)


if __name__ == "__main__":
    main()
