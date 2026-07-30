#!/usr/bin/env python3
"""
Memento 真实数据嵌入对比测试
用 data/testtxt.txt (146条真实问答) 对比 TF-IDF vs Transformer
"""

import json, sys, os, time, re
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = r"D:\工作区\项目\Memento"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from memento.api import Memento

# ─── 解析数据 ───
print("="*70)
print("  Memento 真实数据嵌入对比测试")
print("  数据源: data/testtxt.txt (真实对话)")
print("  TF-IDF+SVD (128维) vs Qwen3-Embedding-0.6B (1024维)")
print("="*70)
print()

with open("data/testtxt.txt", "r", encoding="utf-8-sig") as f:
    raw = f.read()

parts = raw.split("【用户提问】")
parts = [p.strip() for p in parts if p.strip()]
entries = []
for p in parts:
    if "【AI 回答】" in p:
        q, a = p.split("【AI 回答】", 1)
        q = q.strip()
        a = a.strip()
        # 过滤空回答和过短的条目
        if len(q) < 5 or len(a) < 10:
            continue
        # 每条问答作为一条记忆：用户提问 + AI回答(截取前500字)
        combined = f"用户问: {q}\n回答: {a[:500]}"
        entries.append(combined)

print(f"解析完成: {len(entries)} 条记忆")
for i, e in enumerate(entries[:3]):
    print(f"  [{i}] {e[:80]}...")
print()

# ─── 构建系统 A: TF-IDF ───
print("━"*70)
print("  构建系统 A: TF-IDF + SVD (128维)")
print("━"*70)
t0 = time.time()
sys_a = Memento(embedding_model="tfidf-svd", diffusion_hops=2)
for i, text in enumerate(entries):
    sys_a.add_node(text=text, node_id=f"qa_{i:04d}", importance=0.5 + (i % 10) * 0.05)
sys_a.build_index()

# 建图: 相邻节点建边 + 共享关键词建边
tag_groups = defaultdict(list)
for i, text in enumerate(entries):
    # 简单标签: 提取前20个字的关键词
    first_line = text.split("\n")[0][:30]
    tag_groups[first_line[:4]].append(i)

for nid in sys_a.graph.nodes:
    n = sys_a.graph.get_node(nid)
    if n:
        n.vitality = 1.0

# 时间窗口边 (相邻条目建边)
for i in range(len(entries)):
    for j in range(i+1, min(i+4, len(entries))):
        delta_w = 0.05
        sys_a.graph.add_edge(f"qa_{i:04d}", f"qa_{j:04d}",
                             weight=delta_w, edge_type="cooccurrence")

print(f"  完成! 节点={sys_a.graph.node_count}, 边={sys_a.graph.edge_count}, "
      f"耗时={time.time()-t0:.1f}s")
print()

# ─── 构建系统 B: Transformer ───
MODEL_PATH = r"D:\工作区\项目\Memento\models\Qwen3-Embedding-0.6B"
print("━"*70)
print("  构建系统 B: Qwen3-Embedding-0.6B (1024维)")
print("━"*70)
t0 = time.time()
sys_b = Memento(embedding_model=MODEL_PATH, diffusion_hops=2)
for i, text in enumerate(entries):
    sys_b.add_node(text=text, node_id=f"qa_{i:04d}", importance=0.5 + (i % 10) * 0.05)
sys_b.build_index()

for nid in sys_b.graph.nodes:
    n = sys_b.graph.get_node(nid)
    if n:
        n.vitality = 1.0

for i in range(len(entries)):
    for j in range(i+1, min(i+4, len(entries))):
        delta_w = 0.05
        sys_b.graph.add_edge(f"qa_{i:04d}", f"qa_{j:04d}",
                             weight=delta_w, edge_type="cooccurrence")

print(f"  完成! 节点={sys_b.graph.node_count}, 边={sys_b.graph.edge_count}, "
      f"耗时={time.time()-t0:.1f}s")
print()

# ─── 查询对比 ───
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

print("="*70)
print("  查询对比: TF-IDF vs Qwen3-Embedding")
print("="*70)

for q_idx, query in enumerate(queries, 1):
    print(f"\n{'━'*70}")
    print(f"  查询 {q_idx}: 「{query}」")
    print(f"{'━'*70}")

    # 系统 A
    rag_a = sys_a.query_rag_only(query, k=5)
    diff_a = sys_a.query(query, k=5)

    # 系统 B
    rag_b = sys_b.query_rag_only(query, k=5)
    diff_b = sys_b.query(query, k=5)

    # A 纯 RAG
    print(f"\n  ┌─ A (TF-IDF) 纯RAG ─────────────────────────────────────")
    for i, r in enumerate(rag_a, 1):
        txt = r['text'].replace('\n', ' ')[:65]
        print(f"  │ {i}. (sim={r['score']:.4f}) {txt}")
    print(f"  └───────────────────────────────────────────────────────────")

    # B 纯 RAG
    print(f"\n  ┌─ B (Transformer) 纯RAG ─────────────────────────────────")
    for i, r in enumerate(rag_b, 1):
        txt = r['text'].replace('\n', ' ')[:65]
        print(f"  │ {i}. (sim={r['score']:.4f}) {txt}")
    print(f"  └───────────────────────────────────────────────────────────")

    # A 扩散
    print(f"\n  ┌─ A (TF-IDF) 扩散联想 ──────────────────────────────────")
    for i, r in enumerate(diff_a, 1):
        txt = r['text'].replace('\n', ' ')[:50]
        marker = " *扩散发现" if not any(rr["id"] == r["id"] for rr in rag_a) else ""
        print(f"  │ {i}. (s={r['score']:.4f}) {txt}{marker}")
    print(f"  └───────────────────────────────────────────────────────────")

    # B 扩散
    print(f"\n  ┌─ B (Transformer) 扩散联想 ──────────────────────────────")
    for i, r in enumerate(diff_b, 1):
        txt = r['text'].replace('\n', ' ')[:50]
        marker = " *扩散发现" if not any(rr["id"] == r["id"] for rr in rag_b) else ""
        print(f"  │ {i}. (s={r['score']:.4f}) {txt}{marker}")
    print(f"  └───────────────────────────────────────────────────────────")

    # 对比
    rag_ids_a = set(r["id"] for r in rag_a)
    rag_ids_b = set(r["id"] for r in rag_b)
    diff_ids_a = set(r["id"] for r in diff_a)
    diff_ids_b = set(r["id"] for r in diff_b)

    rag_overlap = len(rag_ids_a & rag_ids_b)
    diff_overlap = len(diff_ids_a & diff_ids_b)

    print(f"\n  RAG结果差异: A独有={len(rag_ids_a - rag_ids_b)}, "
          f"B独有={len(rag_ids_b - rag_ids_a)}, 重叠={rag_overlap}")
    print(f"  扩散结果差异: A独有={len(diff_ids_a - diff_ids_b)}, "
          f"B独有={len(diff_ids_b - diff_ids_a)}, 重叠={diff_overlap}")

print()
print("="*70)
print("  测试完成!")
print("="*70)
