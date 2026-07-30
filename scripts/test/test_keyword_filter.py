"""
关键词边过滤方案对比测试

问题诊断:
  keyatten 提取的关键词满足"关键"和"不常见"，但缺乏"惊奇感"。
  技术类话题（v2rayN, MIDI 等实体词）天然具有高惊奇感，
  而创意/情感类话题的关键词可能是无锚定能力的普通词（如"生活费"出现在宇宙讨论中）。

方案:
  A: 语义交叉过滤 — 关键词边只在两端向量余弦相似度高时才保留
     (关键词增强已有语义连接，而非凭空建新连接)
  B: 惊奇感重定义 — 关键词本身的 cos(kw_vec, text_vec) 落在中间带
     (太相似=主题内常见词，太低=噪声，中间带=既相关又意外)

对比:
  baseline : Q/A + KNN (615 边)
  raw      : + 原始关键词边 (~1277 边)
  A        : + 语义交叉过滤后的关键词边
  B        : + 惊奇感过滤后的关键词边
  A+B      : 两者组合
"""

import sys, time, re
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from memento.api import Memento
from memento.index.keyatten_extractor import MemoryKeywordExtractor


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


def truncate(text, n=65):
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def score_stats(scores):
    if not scores:
        return {}
    arr = np.array(scores)
    return {"spread": float(arr.max() - arr.min()),
            "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


QUERIES = [
    "容器启动时配置丢失怎么排查",
    "钢琴连电脑需要什么线和软件",
    "怎么提高学习效率防止晚上崩盘",
    "为什么流行歌都是情情爱爱",
    "手机传文件到电脑用什么软件",
    "独立游戏开发者要不要学美术",
    "梯子和局域网冲突怎么解决",
    "怎么有效休息不会浪费意志力",
]


# ─── 构建函数 ──────────────────────────────────────────────

def build_qa_knn(memento, pairs, knn_k=3):
    """Q/A 拆分 + KNN 建边"""
    for i, p in enumerate(pairs):
        q_id, a_id = f"q_{i:04d}", f"a_{i:04d}"
        memento.add_node(p["question"], node_id=q_id, tags=["question"])
        memento.add_node(p["answer"][:500], node_id=a_id, tags=["answer"])
    memento.build_index()

    for i in range(len(pairs)):
        memento.activate([f"q_{i:04d}", f"a_{i:04d}"])

    vi = memento.vector_index
    all_ids = [f"q_{i:04d}" for i in range(len(pairs))] + \
              [f"a_{i:04d}" for i in range(len(pairs))]
    for nid in all_ids:
        node = memento.graph.get_node(nid)
        if node is None or node.vector is None:
            continue
        results = vi.search(node.vector, k=knn_k + 1)
        for cand_id, sim in results:
            if cand_id != nid and sim > 0.3:
                memento.link(nid, cand_id, weight=float(sim) * 0.3)


def extract_all_keywords(pairs, top_k=5):
    """提取所有节点的关键词，返回 ext, all_ids, all_texts, node_keywords"""
    ext = MemoryKeywordExtractor(device="cuda", default_top_k=top_k)
    all_ids = [f"q_{i:04d}" for i in range(len(pairs))] + \
              [f"a_{i:04d}" for i in range(len(pairs))]
    all_texts = []
    for i, p in enumerate(pairs):
        all_texts.append(p["question"])
        all_texts.append(p["answer"][:500])

    vocab_size = ext.update_idf(all_texts)
    print(f"  IDF 词表: {vocab_size} 词")

    node_keywords = {}
    for nid, text in zip(all_ids, all_texts):
        kws = ext.extract(text)
        node_keywords[nid] = kws

    return ext, all_ids, all_texts, node_keywords


def build_raw_keyword_edges(memento, node_keywords, max_node_freq=20):
    """原始关键词重叠建边（无过滤）"""
    kw_to_nodes = defaultdict(list)
    for nid, kws in node_keywords.items():
        for kw in kws:
            kw_to_nodes[kw].append(nid)

    edge_set = set()
    count = 0
    for kw, nids in kw_to_nodes.items():
        if len(nids) < 2 or len(nids) > max_node_freq:
            continue
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                pair = tuple(sorted([nids[i], nids[j]]))
                if pair in edge_set:
                    continue
                edge_set.add(pair)
                shared = set(node_keywords[pair[0]]) & set(node_keywords[pair[1]])
                weight = len(shared) * 0.15
                memento.link(pair[0], pair[1], weight=min(weight, 0.6))
                count += 1
    return count


def build_semantic_filtered_keyword_edges(
    memento, node_keywords, min_cos_sim=0.30, max_node_freq=20
):
    """方案 A: 语义交叉过滤
    先按关键词重叠建边，再检查两端向量的余弦相似度。
    只保留 cos(node_i, node_j) >= min_cos_sim 的边。
    关键词增强已有的语义连接，而非凭空建新连接。
    """
    kw_to_nodes = defaultdict(list)
    for nid, kws in node_keywords.items():
        for kw in kws:
            kw_to_nodes[kw].append(nid)

    edge_set = set()
    added, rejected = 0, 0
    for kw, nids in kw_to_nodes.items():
        if len(nids) < 2 or len(nids) > max_node_freq:
            continue
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                pair = tuple(sorted([nids[i], nids[j]]))
                if pair in edge_set:
                    continue
                edge_set.add(pair)

                # 语义检查
                ni = memento.graph.get_node(pair[0])
                nj = memento.graph.get_node(pair[1])
                if ni is None or nj is None or ni.vector is None or nj.vector is None:
                    rejected += 1
                    continue
                cos_sim = float(np.dot(ni.vector, nj.vector))
                if cos_sim < min_cos_sim:
                    rejected += 1
                    continue

                shared = set(node_keywords[pair[0]]) & set(node_keywords[pair[1]])
                weight = len(shared) * 0.15
                memento.link(pair[0], pair[1], weight=min(weight, 0.6))
                added += 1

    return added, rejected


def filter_keywords_by_surprisal(
    memento, node_keywords, all_ids,
    min_sim=0.35, max_sim=0.70
):
    """方案 B: 惊奇感重定义
    对每个关键词，计算 cos(keyword_vec, text_vec)。
      - cos 太高 (> max_sim): 主题内常见词，没有惊奇感
      - cos 太低 (< min_sim): 噪声词，和话题无关
      - 中间带: 既相关又意外 → 真正的"惊奇关键词"
    """
    vi = memento.vector_index

    # 收集所有唯一关键词
    all_kws = list(set(kw for kws in node_keywords.values() for kw in kws))
    print(f"  唯一关键词: {len(all_kws)} 个，批量编码中...")

    # 批量编码关键词向量
    kw_vecs = vi.encode(all_kws, mode="document")
    kw_to_vec = {kw: vec for kw, vec in zip(all_kws, kw_vecs)}

    # 逐节点过滤
    filtered = {}
    total_before = 0
    total_after = 0
    stats = {"too_close": 0, "too_far": 0, "kept": 0}

    for nid in all_ids:
        node = memento.graph.get_node(nid)
        if node is None or node.vector is None:
            filtered[nid] = node_keywords.get(nid, [])
            continue

        kws = node_keywords.get(nid, [])
        total_before += len(kws)
        keep = []
        for kw in kws:
            vec = kw_to_vec.get(kw)
            if vec is None:
                continue
            sim = float(np.dot(node.vector, vec))
            if sim < min_sim:
                stats["too_far"] += 1
            elif sim > max_sim:
                stats["too_close"] += 1
            else:
                keep.append(kw)
                stats["kept"] += 1
        filtered[nid] = keep
        total_after += len(keep)

    print(f"  惊奇感过滤: {total_before} → {total_after} 关键词")
    print(f"    太常见(cos>{max_sim}): {stats['too_close']}, "
          f"太噪声(cos<{min_sim}): {stats['too_far']}, "
          f"保留: {stats['kept']}")

    return filtered, kw_to_vec


# ─── 查询与打印 ────────────────────────────────────────────

def run_queries(memento, queries, k=10, seed_k=20):
    results = {}
    for query in queries:
        results[query] = {"diff": memento.query(query, k=k, seed_k=seed_k)}
    return results


def print_results(config_name, results, memento):
    print(f"\n  ┌─ {config_name} (nodes={memento.stats['total_nodes']}, "
          f"edges={memento.stats['total_edges']}) ─────────")
    for qi, query in enumerate(QUERIES, 1):
        diff = results[query]["diff"]
        st = score_stats([it["score"] for it in diff])
        print(f"  │ [{qi}] 「{query}」  spread={st['spread']:.4f}")
        for it in diff[:2]:
            print(f"  │     (s={it['score']:.4f}) {truncate(it['text'], 55)}")
    print(f"  └{'─' * 60}")


# ─── 主流程 ────────────────────────────────────────────────

def main():
    data_path = str(ROOT / "data" / "testtxt.txt")
    model_path = str(ROOT / "models" / "Qwen3-Embedding-0.6B")

    print(f"解析数据: {data_path}")
    pairs = parse_chat_data(data_path)
    print(f"解析完成: {len(pairs)} 条 Q&A 对\n")

    all_results = {}

    # ═══════════════════════════════════════════════════════
    #  Baseline: Q/A + KNN
    # ═══════════════════════════════════════════════════════
    print(f"{'━' * 72}")
    print(f"  baseline: Q/A + KNN")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    print(f"  节点={m.stats['total_nodes']}, 边={m.stats['total_edges']}")
    all_results["baseline"] = {"results": run_queries(m, QUERIES),
                                "edges": m.stats["total_edges"]}
    print_results("baseline", all_results["baseline"]["results"], m)

    # ═══════════════════════════════════════════════════════
    #  Raw: Q/A + KNN + 原始关键词边
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━' * 72}")
    print(f"  raw: Q/A + KNN + 原始关键词边")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    ext, all_ids, all_texts, node_keywords = extract_all_keywords(pairs)

    # 打印样例
    for nid in all_ids[:6]:
        node = m.graph.get_node(nid)
        print(f"  [{nid}] {truncate(node.text, 40)} → {node_keywords[nid]}")

    raw_count = build_raw_keyword_edges(m, node_keywords)
    print(f"  关键词边: {raw_count}")
    print(f"  节点={m.stats['total_nodes']}, 边={m.stats['total_edges']}")
    all_results["raw"] = {"results": run_queries(m, QUERIES),
                          "edges": m.stats["total_edges"]}
    print_results("raw", all_results["raw"]["results"], m)

    # ═══════════════════════════════════════════════════════
    #  A: 语义交叉过滤
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━' * 72}")
    print(f"  A: 语义交叉过滤 (min_cos=0.30)")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    added, rejected = build_semantic_filtered_keyword_edges(
        m, node_keywords, min_cos_sim=0.30)
    print(f"  关键词边: 通过={added}, 拒绝={rejected}")
    print(f"  节点={m.stats['total_nodes']}, 边={m.stats['total_edges']}")
    all_results["A"] = {"results": run_queries(m, QUERIES),
                        "edges": m.stats["total_edges"],
                        "extra": f"pass={added}, reject={rejected}"}
    print_results("A: 语义交叉过滤", all_results["A"]["results"], m)

    # ═══════════════════════════════════════════════════════
    #  B: 惊奇感重定义
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━' * 72}")
    print(f"  B: 惊奇感过滤 (0.35 < cos < 0.70)")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    filtered_kws, kw_to_vec = filter_keywords_by_surprisal(
        m, node_keywords, all_ids, min_sim=0.35, max_sim=0.70)

    # 打印过滤样例
    print(f"\n  过滤样例:")
    for nid in all_ids[:6]:
        node = m.graph.get_node(nid)
        orig = node_keywords[nid]
        kept = filtered_kws[nid]
        removed = [kw for kw in orig if kw not in kept]
        print(f"  [{nid}] {truncate(node.text, 35)}")
        print(f"    保留: {kept}  移除: {removed}")

    b_count = build_raw_keyword_edges(m, filtered_kws)
    print(f"  关键词边: {b_count}")
    print(f"  节点={m.stats['total_nodes']}, 边={m.stats['total_edges']}")
    all_results["B"] = {"results": run_queries(m, QUERIES),
                        "edges": m.stats["total_edges"]}
    print_results("B: 惊奇感过滤", all_results["B"]["results"], m)

    # ═══════════════════════════════════════════════════════
    #  A+B: 组合
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━' * 72}")
    print(f"  A+B: 惊奇感 + 语义交叉过滤")
    print(f"{'━' * 72}")
    m = Memento(embedding_model=model_path, device="cuda",
                diffusion_hops=2, diffusion_alpha=0.3, diffusion_beta=0.6)
    build_qa_knn(m, pairs)
    ab_added, ab_rejected = build_semantic_filtered_keyword_edges(
        m, filtered_kws, min_cos_sim=0.30)
    print(f"  关键词边: 通过={ab_added}, 拒绝={ab_rejected}")
    print(f"  节点={m.stats['total_nodes']}, 边={m.stats['total_edges']}")
    all_results["AB"] = {"results": run_queries(m, QUERIES),
                          "edges": m.stats["total_edges"],
                          "extra": f"pass={ab_added}, reject={ab_rejected}"}
    print_results("A+B: 组合", all_results["AB"]["results"], m)

    # ═══════════════════════════════════════════════════════
    #  汇总
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'=' * 90}")
    print(f"  汇总")
    print(f"{'=' * 90}")

    config_names = {
        "baseline": "baseline Q/A+KNN",
        "raw": "raw +关键词",
        "A": "A 语义过滤",
        "B": "B 惊奇感",
        "AB": "A+B 组合",
    }
    header = f"  {'查询':<24s}"
    for key in ["baseline", "raw", "A", "B", "AB"]:
        header += f" | {config_names[key]:>14s}"
    print(header)
    print(f"  {'-'*24}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")

    avg_spreads = {k: [] for k in ["baseline", "raw", "A", "B", "AB"]}
    for query in QUERIES:
        row = f"  {query[:24]:<24s}"
        for key in ["baseline", "raw", "A", "B", "AB"]:
            diff = all_results[key]["results"][query]["diff"]
            st = score_stats([it["score"] for it in diff])
            avg_spreads[key].append(st["spread"])
            row += f" | {st['spread']:>14.4f}"
        print(row)

    print(f"  {'-'*24}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    row = f"  {'平均':<24s}"
    for key in ["baseline", "raw", "A", "B", "AB"]:
        row += f" | {np.mean(avg_spreads[key]):>14.4f}"
    print(row)

    row = f"  {'边数':<24s}"
    for key in ["baseline", "raw", "A", "B", "AB"]:
        row += f" | {all_results[key]['edges']:>14d}"
    print(row)

    if "extra" in all_results.get("A", {}):
        row = f"  {'A过滤详情':<24s} | {'':>14s} | {'':>14s} | {all_results['A']['extra']:>14s}"
        print(row)
    if "extra" in all_results.get("AB", {}):
        row = f"  {'A+B过滤详情':<24s} | {'':>14s} | {'':>14s} | {'':>14s} | {'':>14s} | {all_results['AB']['extra']:>14s}"
        print(row)

    # 详细对比：挑变化最大的查询
    detail_queries = ["梯子和局域网冲突怎么解决", "独立游戏开发者要不要学美术",
                      "手机传文件到电脑用什么软件", "容器启动时配置丢失怎么排查"]
    for query in detail_queries:
        print(f"\n{'━' * 90}")
        print(f"  详细: 「{query}」")
        print(f"{'━' * 90}")
        for key, name in [("baseline", "baseline"), ("raw", "raw"),
                          ("A", "A语义"), ("B", "B惊奇"), ("AB", "A+B")]:
            diff = all_results[key]["results"][query]["diff"]
            st = score_stats([it["score"] for it in diff])
            print(f"\n  ┌─ {name} (spread={st['spread']:.4f}) ─────")
            for i, it in enumerate(diff[:5], 1):
                tags = it.get("tags", [])
                tag_str = f" [{','.join(tags)}]" if tags else ""
                print(f"  │ {i}. (s={it['score']:.4f}){tag_str} {truncate(it['text'], 55)}")
            print(f"  └{'─' * 60}")

    print(f"\n{'=' * 90}")
    print(f"  完成!")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
