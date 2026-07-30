"""
惊奇度 vs IDF 分析

核心问题：情感/创意类关键词在 cos 惊奇度上是否突出？
和 IDF 对比，看哪个更能区分"好锚点"和"噪声"。
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

    # 提取关键词 + 惊奇度（用 gte-small-zh）
    res = m.build_keyword_edges(top_k=5, compute_surprisal=True)
    idf = m._keyword_extractor.get_idf()

    # ── 按话题分类分析 ──
    print(f"{'=' * 80}")
    print(f"  惊奇度 vs IDF：情感/创意类 vs 技术/实体类")
    print(f"{'=' * 80}")

    # 挑几组有代表性的节点
    groups = [
        # (node_id, label, topic_type)
        ("q_0000", "小说/宇宙设定", "creative"),
        ("q_0001", "小说/XP/巨变", "creative"),
        ("q_0003", "硬魔法系统", "creative"),
        ("q_0005", "三体/歌者文明", "creative"),
        ("a_0018", "艺术/审美直觉", "creative"),
        ("a_0062", "服务/配置/部署", "emotional"),
        ("q_0040", "落寞/心情", "emotional"),
        ("q_0054", "2025回顾/南京", "emotional"),
        ("q_0010", "钢琴/设备", "tech"),
        ("q_0014", "v2rayN/梯子", "tech"),
        ("q_0025", "手机传文件", "tech"),
        ("q_0030", "学习效率", "tech"),
    ]

    creative_kws = []
    emotional_kws = []
    tech_kws = []

    for nid, label, topic in groups:
        node = m.graph.get_node(nid)
        if node is None:
            continue
        surprisal = m.get_keyword_surprisal(nid)
        kws = m.get_node_keywords(nid)

        print(f"\n  [{nid}] {label} ({topic})")
        print(f"  {'关键词':<14s} | {'惊奇度':>8s} | {'IDF':>8s} | {'IDF排名':>8s}")
        print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        # 按惊奇度排序
        sorted_kws = sorted(surprisal.items(), key=lambda x: x[1], reverse=True)
        for kw, s in sorted_kws:
            idf_val = idf.get(kw, 0) if idf else 0
            # IDF 排名（越高越罕见）
            if idf:
                all_idf_vals = sorted(idf.values(), reverse=True)
                rank = next((i for i, v in enumerate(all_idf_vals) if v <= idf_val), len(all_idf_vals))
            else:
                rank = -1
            print(f"  {kw:<14s} | {s:>8.4f} | {idf_val:>8.2f} | {rank:>8d}")

            # 收集数据
            entry = {"kw": kw, "surprisal": s, "idf": idf_val, "rank": rank}
            if topic == "creative":
                creative_kws.append(entry)
            elif topic == "emotional":
                emotional_kws.append(entry)
            else:
                tech_kws.append(entry)

    # ── 统计对比 ──
    print(f"\n\n{'=' * 80}")
    print(f"  按话题类型的统计")
    print(f"{'=' * 80}")

    for name, data in [("creative 创意/小说", creative_kws),
                       ("emotional 情感/心情", emotional_kws),
                       ("tech 技术/工具", tech_kws)]:
        if not data:
            continue
        s_arr = np.array([d["surprisal"] for d in data])
        idf_arr = np.array([d["idf"] for d in data])
        print(f"\n  {name} ({len(data)} 个关键词):")
        print(f"    惊奇度: 均值={s_arr.mean():.4f}  std={s_arr.std():.4f}  "
              f"min={s_arr.min():.4f}  max={s_arr.max():.4f}")
        print(f"    IDF:    均值={idf_arr.mean():.2f}  std={idf_arr.std():.2f}  "
              f"min={idf_arr.min():.2f}  max={idf_arr.max():.2f}")

    # ── 全局分布对比 ──
    print(f"\n\n{'=' * 80}")
    print(f"  全量关键词：惊奇度 vs IDF 散点 (top 20 高惊奇度)")
    print(f"{'=' * 80}")

    all_data = []
    for nid in m._node_keyword_surprisal:
        for kw, s in m._node_keyword_surprisal[nid].items():
            idf_val = idf.get(kw, 0) if idf else 0
            all_data.append({"kw": kw, "surprisal": s, "idf": idf_val})

    # 按惊奇度排序取 top 20
    all_data.sort(key=lambda x: x["surprisal"], reverse=True)
    print(f"\n  top 20 高惊奇度关键词:")
    print(f"  {'关键词':<14s} | {'惊奇度':>8s} | {'IDF':>8s}")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}")
    for d in all_data[:20]:
        print(f"  {d['kw']:<14s} | {d['surprisal']:>8.4f} | {d['idf']:>8.2f}")

    # 按 IDF 排序取 top 20
    all_data.sort(key=lambda x: x["idf"], reverse=True)
    print(f"\n  top 20 高 IDF 关键词:")
    print(f"  {'关键词':<14s} | {'惊奇度':>8s} | {'IDF':>8s}")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}")
    for d in all_data[:20]:
        print(f"  {d['kw']:<14s} | {d['surprisal']:>8.4f} | {d['idf']:>8.2f}")

    # 相关性
    s_all = np.array([d["surprisal"] for d in all_data])
    idf_all = np.array([d["idf"] for d in all_data])
    corr = np.corrcoef(s_all, idf_all)[0, 1]
    print(f"\n  惊奇度 vs IDF 相关系数: {corr:.4f}")

    print(f"\n{'=' * 80}")
    print(f"  完成!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
