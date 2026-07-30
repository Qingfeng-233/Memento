"""
详细查看 a_0062 节点的文本、关键词、惊奇度
"""
import sys, re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memento.api import Memento


def parse_chat_data(filepath):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()
    parts = re.split(r'【用户提问】', content)
    pairs = []
    for part in parts:
        part = part.strip()
        if not part or '【AI 回答】' not in part:
            continue
        q, a = part.split('【AI 回答】', 1)
        q, a = q.strip(), a.strip()
        if q:
            pairs.append({"question": q, "answer": a})
    return pairs


def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")
    pairs = parse_chat_data(data_path)

    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)

    for i, p in enumerate(pairs):
        q_id, a_id = f"q_{i:04d}", f"a_{i:04d}"
        m.add_node(p["question"], node_id=q_id, tags=["question"])
        m.add_node(p["answer"][:500], node_id=a_id, tags=["answer"])
    m.build_index()

    # 提取关键词 + 惊奇度
    res = m.build_keyword_edges(top_k=5, compute_surprisal=True)

    # 查看 a_0062
    nid = "a_0062"
    node = m.graph.get_node(nid)
    kws = m.get_node_keywords(nid)
    surprisal = m.get_keyword_surprisal(nid)

    print(f"{'=' * 70}")
    print(f"  节点 {nid}")
    print(f"{'=' * 70}")

    print(f"\n  ── 原始回答文本（截断到 500 字）──")
    print(f"  {node.text}")

    print(f"\n  ── 对应的提问（q_0062）──")
    q_node = m.graph.get_node("q_0062")
    if q_node:
        print(f"  {q_node.text}")

    print(f"\n  ── 关键词 + 惊奇度（降序）──")
    sorted_kws = sorted(surprisal.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'关键词':<16s} | {'惊奇度':>8s}")
    print(f"  {'-'*16}-+-{'-'*8}")
    for kw, s in sorted_kws:
        print(f"  {kw:<16s} | {s:>8.4f}")

    # 再多看几个节点，让用户感受整体分布
    print(f"\n\n{'=' * 70}")
    print(f"  其他几个节点对比")
    print(f"{'=' * 70}")
    for check_nid in ["q_0000", "q_0005", "a_0018", "q_0054", "q_0040"]:
        c_node = m.graph.get_node(check_nid)
        if c_node is None:
            continue
        c_kws = m.get_node_keywords(check_nid)
        c_surprisal = m.get_keyword_surprisal(check_nid)
        sorted_c = sorted(c_surprisal.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  [{check_nid}] 文本前80字: {c_node.text[:80]}...")
        print(f"  {'关键词':<16s} | {'惊奇度':>8s}")
        print(f"  {'-'*16}-+-{'-'*8}")
        for kw, s in sorted_c:
            print(f"  {kw:<16s} | {s:>8.4f}")


if __name__ == "__main__":
    main()
