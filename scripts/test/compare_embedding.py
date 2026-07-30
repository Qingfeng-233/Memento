#!/usr/bin/env python3
"""
Memento 嵌入升级对比测试

对比 TF-IDF+SVD vs Sentence-Transformer (paraphrase-multilingual-MiniLM-L12-v2)
展示两者在相同查询下的不同表现
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\工作区\项目\Memento")

from memento.api import Memento


def load_memories():
    """加载测试数据"""
    data_path = Path(r"D:\工作区\项目\Memento\memories.jsonl")
    memories = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                memories.append(json.loads(line))
    return memories


def build_graph(mem: Memento, memories: list):
    """构建情境共现图"""
    # 基于主题标签的共现
    tag_groups = defaultdict(list)
    for m in memories:
        tags = m.get("tags", [])
        if tags:
            tag_groups[tags[0]].append(m)

    for tag, group in tag_groups.items():
        group.sort(key=lambda x: x.get("created_at", ""))
        window_size = 4
        for i in range(len(group)):
            for j in range(i + 1, min(i + window_size, len(group))):
                m_a, m_b = group[i], group[j]
                n_a = mem.graph.get_node(m_a["id"])
                n_b = mem.graph.get_node(m_b["id"])
                if n_a and n_b:
                    delta_w = 0.08 * n_a.vitality * n_b.vitality
                    mem.graph.add_edge(m_a["id"], m_b["id"],
                                       weight=delta_w,
                                       edge_type="cooccurrence")

    # 基于时间窗口的共现
    sorted_mems = sorted(memories, key=lambda x: x.get("created_at", ""))
    window_size = 5
    for i in range(len(sorted_mems)):
        for j in range(i + 1, min(i + window_size, len(sorted_mems))):
            m_a, m_b = sorted_mems[i], sorted_mems[j]
            tags_a = set(m_a.get("tags", []))
            tags_b = set(m_b.get("tags", []))
            shared = len(tags_a & tags_b)
            base_w = 0.03 + 0.02 * shared
            mem.graph.add_edge(m_a["id"], m_b["id"],
                               weight=base_w,
                               edge_type="cooccurrence")


def init_system(embedding_model: str, memories: list) -> Memento:
    """初始化 Memento 系统"""
    mem = Memento(embedding_model=embedding_model, diffusion_hops=2)

    for m in memories:
        mem.add_node(
            text=m["text"],
            node_id=m["id"],
            importance=m.get("importance", 0.5),
            tags=m.get("tags", []),
            source=m.get("source", "import"),
            created_at=m.get("created_at"),
        )

    mem.build_index()
    build_graph(mem, memories)
    return mem


def run_comparison():
    """运行对比测试"""

    print("="*70)
    print("  Memento 嵌入升级对比测试")
    print("  TF-IDF+SVD (128维) vs Sentence-Transformer (384维)")
    print("="*70)
    print()

    memories = load_memories()
    print(f"数据量: {len(memories)} 条记忆")
    print()

    # ─── 系统 A: TF-IDF+SVD ───
    print("━" * 70)
    print("  构建系统 A: TF-IDF + SVD (128维)")
    print("━" * 70)
    t0 = time.time()
    sys_a = init_system("tfidf-svd", memories)
    print(f"  完成! 节点={sys_a.graph.node_count}, 边={sys_a.graph.edge_count}, "
          f"耗时={time.time()-t0:.1f}s")
    print(f"  向量维度: {sys_a.vector_index.dimension}")
    print()

    # ─── 系统 B: Sentence-Transformer ───
    print("━" * 70)
    print("  构建系统 B: Sentence-Transformer (384维)")
    print("  模型: paraphrase-multilingual-MiniLM-L12-v2")
    print("━" * 70)
    t0 = time.time()
    sys_b = init_system(
        r"C:\Users\29864\.cache\modelscope\hub\models\sentence-transformers\paraphrase-multilingual-MiniLM-L12-v2",
        memories
    )
    print(f"  完成! 节点={sys_b.graph.node_count}, 边={sys_b.graph.edge_count}, "
          f"耗时={time.time()-t0:.1f}s")
    print(f"  向量维度: {sys_b.vector_index.dimension}")
    print()

    # ─── 查询对比 ───
    queries = [
        "深度学习模型压缩与优化",
        "自然语言处理的应用",
        "推荐系统与知识图谱",
        "如何学习人工智能",
        "联邦学习隐私保护",
    ]

    print("="*70)
    print("  查询对比")
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

        # 显示结果
        print(f"\n  ┌─ 系统A (TF-IDF) 纯RAG ─────────────────────────────────")
        for i, r in enumerate(rag_a, 1):
            print(f"  │ {i}. (sim={r['score']:.4f}) {r['text'][:55]}")
        print(f"  └───────────────────────────────────────────────────────────")

        print(f"\n  ┌─ 系统A (TF-IDF) 扩散联想 ──────────────────────────────")
        for i, r in enumerate(diff_a, 1):
            tag = "  *扩散发现" if not any(rr["id"] == r["id"] for rr in rag_a) else ""
            print(f"  │ {i}. (s={r['score']:.4f}, ω={r['importance']:.2f}, "
                  f"v={r['vitality']:.2f}) {r['text'][:40]}{tag}")
        print(f"  └───────────────────────────────────────────────────────────")

        print(f"\n  ┌─ 系统B (Transformer) 纯RAG ─────────────────────────────")
        for i, r in enumerate(rag_b, 1):
            print(f"  │ {i}. (sim={r['score']:.4f}) {r['text'][:55]}")
        print(f"  └───────────────────────────────────────────────────────────")

        print(f"\n  ┌─ 系统B (Transformer) 扩散联想 ──────────────────────────")
        for i, r in enumerate(diff_b, 1):
            tag = "  *扩散发现" if not any(rr["id"] == r["id"] for rr in rag_b) else ""
            print(f"  │ {i}. (s={r['score']:.4f}, ω={r['importance']:.2f}, "
                  f"v={r['vitality']:.2f}) {r['text'][:40]}{tag}")
        print(f"  └───────────────────────────────────────────────────────────")

        # 对比统计
        ids_a = set(r["id"] for r in diff_a)
        ids_b = set(r["id"] for r in diff_b)
        only_a = ids_a - ids_b
        only_b = ids_b - ids_a
        common = ids_a & ids_b

        print(f"\n  扩散结果差异:")
        print(f"    A独有 (TF-IDF): {len(only_a)} 条")
        print(f"    B独有 (Transformer): {len(only_b)} 条")
        print(f"    两者都有: {len(common)} 条")

        if only_b:
            print(f"\n  Transformer 扩散发现的独特记忆:")
            for nid in list(only_b)[:3]:
                node = sys_b.graph.get_node(nid)
                if node:
                    print(f"    [{nid}] {node.text[:60]}")

    # ─── 总结 ───
    print()
    print("="*70)
    print("  对比总结")
    print("="*70)
    print()
    print("  TF-IDF+SVD (128维):")
    print("    - 基于关键词频率统计，语义理解较弱")
    print("    - 同义词/近义词可能无法召回")
    print("    - 适合演示系统架构，生产环境不够")
    print()
    print("  Sentence-Transformer (384维):")
    print("    - 基于预训练语义模型，理解语义关系")
    print("    - 能召回不同措辞但同义的内容")
    print("    - 扩散效果更好，因为种子向量质量更高")
    print()
    print("  下一步优化方向:")
    print("    - 使用更大的中文模型 (如 bge-large-zh)")
    print("    - 增加真实用户交互数据构建边")
    print("    - 调参扩散系数和衰减率")
    print()
    print("="*70)
    print("  测试完成!")
    print("="*70)


if __name__ == "__main__":
    run_comparison()
