"""验证新 energy 公式：USB/引入软件 压低，LocalSend 保持，情感词豁免。"""
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

    # 重点关注的词
    targets = ["USB", "引入软件", "引入软件确实", "软件",
               "LocalSend", "Syncthing", "MIDI to USB",
               "开心", "难过", "孤独", "愧疚", "落寞", "内耗", "羁绊",
               "电脑扫描手机", "发现钢琴", "v2rayN"]

    print(f"{'概念':<16} {'df':>3} {'generic':>8} {'emotion':>7} "
          f"{'energy':>7}  判断")
    print("-" * 70)
    for t in targets:
        cid = cg.concept_id(t)
        c = cg.concepts.get(cid)
        if c is None:
            print(f"{t:<16}  (不在概念图里 — 可能被过滤了)")
            continue
        # 重新算 genericness 和 emotion（用于展示）
        # 注意：genericness 没存到 ConceptNode，只存了 final energy
        is_emo = ConceptGraph.is_emotion_word(t)
        print(f"{t:<16} {c.doc_freq:>3} {'?':>8} "
              f"{'是' if is_emo else '否':>7} {c.initial_energy:>7.4f}  "
              f"{'★情感豁免' if is_emo else ''}")

    # 对比关键对
    print(f"\n{'='*70}")
    print("关键对比（新公式 vs 旧公式的体感差异）")
    print(f"{'='*70}")

    pairs_check = [
        ("USB", "应被压低（跨主题泛词）"),
        ("软件", "应被压低（通用实词）"),
        ("引入软件确实", "应被压低或过滤（垃圾短语）"),
        ("LocalSend", "应保持高分（专有名词）"),
        ("Syncthing", "应保持高分（专有名词）"),
        ("v2rayN", "应保持高分（专有名词）"),
        ("开心", "应豁免保持高分（情感词）"),
        ("孤独", "应豁免保持高分（情感词）"),
        ("愧疚", "应豁免保持高分（情感词）"),
    ]
    for kw, expected in pairs_check:
        cid = cg.concept_id(kw)
        c = cg.concepts.get(cid)
        if c is None:
            print(f"  {kw:<14} 被过滤了（energy < 0.5） — {expected}")
        else:
            status = "✓" if (c.initial_energy >= 0.5) else "✗ 压低了"
            print(f"  {kw:<14} energy={c.initial_energy:.4f} {status} — {expected}")


if __name__ == "__main__":
    main()
