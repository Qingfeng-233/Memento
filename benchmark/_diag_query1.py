"""诊断查询1'容器启动时配置丢失怎么排查'的排序异常"""
import sys, json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from benchmark.compare_memory_systems import (
    MementoAdapter, parse_chat_data, memory_text,
)

# 加载数据 + 构建（和盲测完全一样的参数）
pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
print(f"data: {len(pairs)} Q&A")

adapter = MementoAdapter(
    top_k=10,
    use_concepts=True,
    embedding_model="api:Qwen/Qwen3-Embedding-4B",
)
print("building Memento...")
info = adapter.build(pairs)
print(f"build OK: indexed={info['indexed']}, concept_seconds={info['concept_seconds']:.1f}s")

# 用 debug=True 查询
query = "容器启动时配置丢失怎么排查"
print(f"\n{'='*60}")
print(f"Query: {query}")
print(f"{'='*60}")

engine = adapter.engine
result = engine.query_with_concepts(
    query,
    k=10,
    seed_k=max(20, 10),
    concept_k=10,
    concept_hops=2,
    concept_weight=0.45,
    debug=True,
    debug_top_concepts=15,
)

# 打印种子关键词
print(f"\n--- Seed Concepts (top 15) ---")
for sc in result["seed_concepts"]:
    print(f"  [{sc['activation']:.4f}] sim={sc['similarity']:.4f} "
          f"energy={sc['initial_energy']:.4f} df={sc['doc_freq']} "
          f"'{sc['concept']}'")

# 打印扩散后激活的关键词
print(f"\n--- Activated Concepts (top 15) ---")
for ac in result["activated_concepts"]:
    print(f"  [{ac['activation']:.4f}] energy={ac['initial_energy']:.4f} "
          f"df={ac['doc_freq']} '{ac['concept']}'")

# 打印结果（拆开 rag_score 和 concept_score）
print(f"\n--- Results (top 10) ---")
print(f"{'Rank':<5} {'Final':>7} {'RAG':>7} {'Concept':>8} {'C_Weight':>9} "
      f"{'ID':<12} {'Text (first 80 chars)'}")
print("-" * 130)
for i, hit in enumerate(result["results"], 1):
    final = hit["score"]
    rag = hit["rag_score"]
    concept = hit["concept_score"]
    cw = 0.45 * concept
    text_preview = hit["text"][:80].replace("\n", " ")
    print(f"{i:<5} {final:>7.4f} {rag:>7.4f} {concept:>8.4f} {cw:>9.4f} "
          f"{hit['id']:<12} {text_preview}")

    # 打印 concept_supports
    if hit.get("concept_supports"):
        for cs in hit["concept_supports"][:5]:
            print(f"        support: '{cs.get('concept','?')}' score={cs.get('score',0):.4f}")

# 也跑纯 RAG（不用概念图）做对比
print(f"\n{'='*60}")
print("Pure RAG (no concept graph)")
print(f"{'='*60}")

engine2 = adapter.engine
query_vector = engine2.vector_index.encode([query], mode="query")[0]
rag_hits = engine2.vector_index.search(query_vector, k=10)
print(f"\n{'Rank':<5} {'Score':>7} {'ID':<12} {'Text (first 80 chars)'}")
print("-" * 110)
for i, (node_id, score) in enumerate(rag_hits, 1):
    node = engine2.graph.get_node(node_id)
    text_preview = node.text[:80].replace("\n", " ")
    print(f"{i:<5} {score:>7.4f} {node_id:<12} {text_preview}")
