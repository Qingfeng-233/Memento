#!/usr/bin/env python3
"""
Memento 三系统对比测试 (V2)
===============================
  A: TF-IDF + SVD (128维) — 关键词精确匹配
  B: Qwen3-Embedding (1024维, instruction prefix 修正) — 语义检索
  C: Qwen3 + 惊奇关键词增强图 — 语义 + 关键词锚点扩散

测试数据: data/testtxt.txt (143条真实对话)
"""

import json, sys, os, time, re
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = r"D:\工作区\项目\Memento"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from memento.api import Memento
from memento.index.keyword_extractor import KeywordExtractor

# ═══════════════════════════════════════════════════════════
# 0. 解析数据
# ═══════════════════════════════════════════════════════════
print("=" * 72)
print("  Memento 三系统对比测试 (V2)")
print("  数据源: data/testtxt.txt (真实对话)")
print("  A: TF-IDF+SVD | B: Qwen3(前缀修正) | C: Qwen3+惊奇关键词")
print("=" * 72)
print()

with open("data/testtxt.txt", "r", encoding="utf-8-sig") as f:
    raw = f.read()

parts = raw.split("【用户提问】")
parts = [p.strip() for p in parts if p.strip()]
entries = []
for p in parts:
    if "【AI 回答】" in p:
        q, a = p.split("【AI 回答】", 1)
        q, a = q.strip(), a.strip()
        if len(q) < 5 or len(a) < 10:
            continue
        combined = f"用户问: {q}\n回答: {a[:500]}"
        entries.append(combined)

print(f"解析完成: {len(entries)} 条记忆")

# ═══════════════════════════════════════════════════════════
# 1. 构建系统 A: TF-IDF
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 72)
print("  系统 A: TF-IDF + SVD (128维)")
print("━" * 72)
t0 = time.time()
sys_a = Memento(embedding_model="tfidf-svd", diffusion_hops=2)
for i, text in enumerate(entries):
    sys_a.add_node(text=text, node_id=f"qa_{i:04d}",
                   importance=0.5 + (i % 10) * 0.05)
sys_a.build_index()
# 时间窗口边
for nid in sys_a.graph.nodes:
    n = sys_a.graph.get_node(nid)
    if n:
        n.vitality = 1.0
for i in range(len(entries)):
    for j in range(i + 1, min(i + 4, len(entries))):
        sys_a.graph.add_edge(f"qa_{i:04d}", f"qa_{j:04d}",
                             weight=0.05, edge_type="cooccurrence")
t_a = time.time() - t0
print(f"  完成! 节点={sys_a.graph.node_count}, 边={sys_a.graph.edge_count}, "
      f"耗时={t_a:.1f}s")

# ═══════════════════════════════════════════════════════════
# 2. 构建系统 B: Qwen3 (prefix 修正)
# ═══════════════════════════════════════════════════════════
MODEL_PATH = r"D:\工作区\项目\Memento\models\Qwen3-Embedding-0.6B"
print("\n" + "━" * 72)
print("  系统 B: Qwen3-Embedding (1024维, 前缀修正)")
print("━" * 72)
t0 = time.time()
sys_b = Memento(embedding_model=MODEL_PATH, diffusion_hops=2)
for i, text in enumerate(entries):
    sys_b.add_node(text=text, node_id=f"qa_{i:04d}",
                   importance=0.5 + (i % 10) * 0.05)
sys_b.build_index()
for nid in sys_b.graph.nodes:
    n = sys_b.graph.get_node(nid)
    if n:
        n.vitality = 1.0
for i in range(len(entries)):
    for j in range(i + 1, min(i + 4, len(entries))):
        sys_b.graph.add_edge(f"qa_{i:04d}", f"qa_{j:04d}",
                             weight=0.05, edge_type="cooccurrence")
t_b = time.time() - t0
print(f"  完成! 节点={sys_b.graph.node_count}, 边={sys_b.graph.edge_count}, "
      f"耗时={t_b:.1f}s")

# ═══════════════════════════════════════════════════════════
# 3. 构建系统 C: Qwen3 + 惊奇关键词增强
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 72)
print("  系统 C: Qwen3 + 惊奇关键词增强图")
print("━" * 72)
t0 = time.time()

# 3a. 初始化关键词提取器（统计方法：TF-IDF × 稀有度 × 具体性）
extractor = KeywordExtractor(method="statistical")
extractor.fit_corpus(entries)

# 3b. 提取每个节点的惊奇关键词
print("  提取各节点的惊奇关键词 ...")
node_keywords = []
node_ids = [f"qa_{i:04d}" for i in range(len(entries))]
for i, text in enumerate(entries[:10]):
    kws = extractor.extract(text, top_k=5)
    node_keywords.append(kws)
    if i < 3:
        print(f"  [{i}] {', '.join(f'{w}({s:.2f})' for w, s in kws)}")
# 剩余批量
for i, text in enumerate(entries[10:], 10):
    kws = extractor.extract(text, top_k=5)
    node_keywords.append(kws)

# 3c. 构建独立系统 C: Qwen3 + 惊奇关键词图增强
sys_c = Memento(embedding_model=MODEL_PATH, diffusion_hops=2)
for i, text in enumerate(entries):
    sys_c.add_node(text=text, node_id=f"qa_{i:04d}",
                   importance=0.5 + (i % 10) * 0.05)
sys_c.build_index()
for nid in sys_c.graph.nodes:
    n = sys_c.graph.get_node(nid)
    if n:
        n.vitality = 1.0

# 时间窗口边 + 惊奇关键词重叠边
for i in range(len(entries)):
    for j in range(i + 1, min(i + 4, len(entries))):
        sys_c.graph.add_edge(f"qa_{i:04d}", f"qa_{j:04d}",
                             weight=0.05, edge_type="cooccurrence")

# 惊奇关键词重叠边
kw_edges = extractor.build_keyword_graph(node_keywords, node_ids, min_overlap=1)
for src, tgt, weight, shared in kw_edges:
    sys_c.graph.add_edge(src, tgt, weight=weight * 0.15,
                         edge_type="keyword_surprise")

t_c = time.time() - t0
print(f"  惊奇关键词重叠边: {len(kw_edges)} 条")
print(f"  完成! 节点={sys_c.graph.node_count}, 边={sys_c.graph.edge_count}, "
      f"耗时={t_c:.1f}s")

# ═══════════════════════════════════════════════════════════
# 4. 查询对比
# ═══════════════════════════════════════════════════════════
queries = [
    "容器启动时配置丢失怎么排查",
    "钢琴连电脑需要什么线和软件",
    "怎么提高学习效率防止晚上崩盘",
    "为什么流行歌都是情情爱爱",
    "手机传文件到电脑用什么软件",
    "独立游戏开发者要不要学美术",
    "梯子和局域网冲突怎么解决",
    "怎么有效休息不会浪费意志力",
]

print("\n" + "=" * 72)
print("  查询对比: A(TF-IDF) vs B(Qwen3修正) vs C(Qwen3+惊奇关键词)")
print("=" * 72)

# 统计
total_rag_diff = {"A-B": 0, "A-C": 0, "B-C": 0}

for q_idx, query in enumerate(queries, 1):
    print(f"\n{'━' * 72}")
    print(f"  查询 {q_idx}: 「{query}」")
    print(f"{'━' * 72}")

    # ─── 系统 A ───
    print(f"\n  ┌─ A (TF-IDF) 纯RAG ─────────────────────────────────────")
    rag_a = sys_a.query_rag_only(query, k=5)
    for i, r in enumerate(rag_a, 1):
        txt = r['text'].replace('\n', ' ')[:60]
        print(f"  │ {i}. (sim={r['score']:.4f}) {txt}")
    print(f"  └───────────────────────────────────────────────────────")

    # ─── 系统 B ───
    print(f"\n  ┌─ B (Qwen3 前缀修正) 纯RAG ────────────────────────────")
    rag_b = sys_b.query_rag_only(query, k=5)
    for i, r in enumerate(rag_b, 1):
        txt = r['text'].replace('\n', ' ')[:60]
        print(f"  │ {i}. (sim={r['score']:.4f}) {txt}")
    print(f"  └───────────────────────────────────────────────────────")

    # ─── 系统 C ───
    print(f"\n  ┌─ C (Qwen3+惊奇关键词) 纯RAG ─────────────────────────")
    rag_c = sys_c.query_rag_only(query, k=5)
    for i, r in enumerate(rag_c, 1):
        txt = r['text'].replace('\n', ' ')[:60]
        print(f"  │ {i}. (sim={r['score']:.4f}) {txt}")
    print(f"  └───────────────────────────────────────────────────────")

    # ─── 扩散对比 ───
    diff_a = sys_a.query(query, k=5)
    diff_b = sys_b.query(query, k=5)
    diff_c = sys_c.query(query, k=5)

    print(f"\n  ┌─ A (TF-IDF) 扩散联想 ──────────────────────────────")
    for i, r in enumerate(diff_a, 1):
        txt = r['text'].replace('\n', ' ')[:45]
        marker = " *扩散" if not any(rr["id"] == r["id"] for rr in rag_a) else ""
        print(f"  │ {i}. (s={r['score']:.4f}) {txt}{marker}")
    print(f"  └─────────────────────────────────────────────────────")

    print(f"\n  ┌─ B (Qwen3) 扩散联想 ────────────────────────────────")
    for i, r in enumerate(diff_b, 1):
        txt = r['text'].replace('\n', ' ')[:45]
        marker = " *扩散" if not any(rr["id"] == r["id"] for rr in rag_b) else ""
        print(f"  │ {i}. (s={r['score']:.4f}) {txt}{marker}")
    print(f"  └─────────────────────────────────────────────────────")

    print(f"\n  ┌─ C (Qwen3+惊奇关键词) 扩散联想 ─────────────────────")
    for i, r in enumerate(diff_c, 1):
        txt = r['text'].replace('\n', ' ')[:45]
        marker = " *扩散" if not any(rr["id"] == r["id"] for rr in rag_c) else ""
        print(f"  │ {i}. (s={r['score']:.4f}) {txt}{marker}")
    print(f"  └─────────────────────────────────────────────────────")

    # ─── 对比统计 ───
    rag_ids_a = set(r["id"] for r in rag_a)
    rag_ids_b = set(r["id"] for r in rag_b)
    rag_ids_c = set(r["id"] for r in rag_c)
    diff_ids_a = set(r["id"] for r in diff_a)
    diff_ids_b = set(r["id"] for r in diff_b)
    diff_ids_c = set(r["id"] for r in diff_c)

    print(f"\n  RAG重叠: A∩B={len(rag_ids_a & rag_ids_b)}, "
          f"A∩C={len(rag_ids_a & rag_ids_c)}, "
          f"B∩C={len(rag_ids_b & rag_ids_c)}")
    print(f"  扩散重叠: A∩B={len(diff_ids_a & diff_ids_b)}, "
          f"A∩C={len(diff_ids_a & diff_ids_c)}, "
          f"B∩C={len(diff_ids_b & diff_ids_c)}")

    # 扩散新发现（不在任何 RAG 结果中的项）
    all_rag_ids = rag_ids_a | rag_ids_b | rag_ids_c
    new_a = diff_ids_a - all_rag_ids
    new_b = diff_ids_b - all_rag_ids
    new_c = diff_ids_c - all_rag_ids
    print(f"  扩散新发现(不在任何RAG中): A={len(new_a)}, B={len(new_b)}, C={len(new_c)}")

    # ─── 惊奇关键词展示 ───
    print(f"\n  ◆ 查询惊奇关键词: ", end="")
    q_kws = extractor.extract_for_query(query, top_k=5)
    print(", ".join(f"{w}({s:.2f})" for w, s in q_kws))

print("\n" + "=" * 72)
print("  测试完成!")
print(f"  耗时: A={t_a:.1f}s  B={t_b:.1f}s  C={t_c:.1f}s")
print("=" * 72)
