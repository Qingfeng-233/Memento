"""查手机传文件查询激活的概念，和 qa_0014 连的概念，在向量空间的余弦距离。"""
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

    # 1. 查询激活了哪些概念（种子）
    qvec = engine.vector_index.encode([query], mode="query")[0]
    q = qvec.astype(np.float32).reshape(1, -1)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn

    concept_items = list(cg.concepts.items())
    concept_ids = [cid for cid, _ in concept_items]
    concept_vecs = np.vstack([c.vector for _, c in concept_items])
    sims_q = (concept_vecs @ q.T).reshape(-1)

    # 种子 top-10
    print(f"查询：「{query}」\n")
    print("=== 查询激活的种子概念 top-12 ===")
    seed_order = np.argsort(-sims_q)[:12]
    seed_set = set()
    for idx in seed_order:
        cid = concept_ids[int(idx)]
        c = cg.concepts[cid]
        sim = float(sims_q[int(idx)])
        print(f"  {c.text:<20} cos(查询,概念)={sim:.4f}")
        seed_set.add(cid)

    # 2. qa_0014 连着哪些概念
    print(f"\n=== qa_0014（这玩意能买吗 MIDI设备）连接的概念 ===")
    target = "qa_0014"
    target_concepts = cg.event_to_concepts.get(target, {})
    for cid, w in sorted(target_concepts.items(), key=lambda x: -x[1]):
        c = cg.concepts.get(cid)
        if c:
            in_seed = "★种子" if cid in seed_set else ""
            print(f"  {c.text:<20} event边权={w:.3f}  {in_seed}")

    # 3. 关键：种子概念 和 qa_0014 的概念 之间的余弦
    print(f"\n=== 种子概念 vs qa_0014概念的 余弦距离 ===")
    print(f"{'种子概念':<20} {'qa_0014概念':<20} {'cos':>7}")
    print("-"*50)
    for scid in list(seed_set)[:8]:
        s_text = cg.concepts[scid].text
        s_vec = cg.concepts[scid].vector
        for tcid, tw in target_concepts.items():
            t_vec = cg.concepts[tcid].vector
            cos = float(np.dot(s_vec, t_vec))
            t_text = cg.concepts[tcid].text
            if cos > 0.3:  # 只看近的
                print(f"{s_text:<20} {t_text:<20} {cos:.4f}")

    # 4. 直接看：查询向量 vs qa_0014 的文本向量
    qa14 = engine.graph.get_node("qa_0014")
    if qa14 and qa14.vector is not None:
        cos_direct = float(np.dot(qvec, qa14.vector))
        print(f"\n=== 查询 vs qa_0014 文本的直接余弦 ===")
        print(f"  cos = {cos_direct:.4f}")
        print(f"  （这是纯 RAG 分数的基础）")

    # 5. 对比：查询 vs qa_0024（手机传文件本身的正确答案）
    qa24 = engine.graph.get_node("qa_0024")
    if qa24 and qa24.vector is not None:
        cos_24 = float(np.dot(qvec, qa24.vector))
        print(f"\n  对比：查询 vs qa_0024（正确答案）cos = {cos_24:.4f}")


if __name__ == "__main__":
    main()
