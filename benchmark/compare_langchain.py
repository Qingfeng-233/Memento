"""
Memento vs LangChain (标准 RAG) 对比测试

LangChain 代表"标准向量 RAG 管道":
  - 同一份数据写入 FAISS
  - 查询时做 top-k 余弦检索
  - 没有概念图、没有扩散、没有上下文路由

两个 LangChain 变体:
  1. LangChain-SF: SiliconFlow Qwen3-4B embedding（与 Mem0/Letta 同模型）
  2. LangChain-Local: 本地 Qwen3-0.6B embedding（与 Memento 同模型）

这样能区分"算法优势"和"embedding 模型优势"。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"

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


# ──────────────────── 工具函数 ────────────────────

def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def parse_chat_data(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    content = path.read_text(encoding="utf-8-sig")
    pairs: list[dict[str, str]] = []
    for part in re.split(r"【用户提问】", content):
        part = part.strip()
        if not part or "【AI 回答】" not in part:
            continue
        question, answer = part.split("【AI 回答】", 1)
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def memory_text(pair: dict[str, str]) -> str:
    return f"用户问: {pair['question']}\n回答: {pair['answer']}"


def truncate(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "..." if len(text) > limit else text


@dataclass
class Hit:
    text: str
    score: float | None = None


# ──────────────────── Memento 适配器 ────────────────────

class MementoAdapter:
    name = "Memento"

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.engine = None

    def build(self, pairs: list[dict[str, str]]) -> dict[str, Any]:
        from memento.api import Memento

        self.engine = Memento(
            embedding_model=str(ROOT / "models" / "Qwen3-Embedding-0.6B"),
            diffusion_hops=2,
        )
        t0 = time.time()
        for idx, pair in enumerate(pairs):
            self.engine.add_node(
                text=memory_text(pair),
                node_id=f"qa_{idx:04d}",
                importance=0.5,
            )
        self.engine.build_index()
        index_sec = time.time() - t0

        t1 = time.time()
        info = self.engine.build_concept_graph(
            top_k=8,
            keyword_model=str(ROOT / "models" / "Qwen3-Embedding-0.6B"),
            keyword_dtype="float16",
            keyword_cache_dir="data/keyatten_cache",
            keyword_sim_threshold=0.45,
            keyword_temperature=0.06,
            keyword_top_neighbors=8,
            max_concepts=500,
            min_concept_energy=0.5,
        )
        concept_sec = time.time() - t1

        return {
            "index_seconds": round(index_sec, 1),
            "concept_seconds": round(concept_sec, 1),
            "concepts": info["concepts"],
            "concept_edges": info["concept_edges"],
            "context_vectors": len(self.engine.concept_graph.context_vectors),
        }

    def search(self, query: str) -> tuple[list[Hit], float]:
        t0 = time.time()
        raw = self.engine.query_with_concepts(
            query, k=self.top_k, seed_k=20, concept_k=10,
            concept_hops=2, concept_weight=0.45,
        )
        elapsed = time.time() - t0
        hits = [
            Hit(text=item["text"], score=float(item.get("score", 0.0)))
            for item in raw
        ]
        return hits, elapsed


# ──────────────────── LangChain FAISS 适配器 ────────────────────

class LangChainFAISSAdapter:
    """标准 RAG: LangChain + FAISS，无概念图，无扩散。"""

    def __init__(self, top_k: int, embedding_backend: str = "siliconflow") -> None:
        """
        embedding_backend:
          - "siliconflow": SiliconFlow Qwen3-4B (与 Mem0/Letta 对齐)
          - "local": 本地 Qwen3-0.6B (与 Memento 对齐)
        """
        self.top_k = top_k
        self.embedding_backend = embedding_backend
        self.vectorstore = None
        if embedding_backend == "siliconflow":
            self.name = "LangChain-SF (4B)"
        else:
            self.name = "LangChain-Local (0.6B)"

    def build(self, pairs: list[dict[str, str]]) -> dict[str, Any]:
        from langchain_community.vectorstores import FAISS

        texts = [memory_text(pair) for pair in pairs]
        metadatas = [{"idx": i} for i in range(len(pairs))]

        if self.embedding_backend == "siliconflow":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                model="Qwen/Qwen3-Embedding-4B",
                openai_api_key=require_env("SILICONFLOW_API_KEY"),
                openai_api_base=require_env("SILICONFLOW_API_BASE"),
                dimensions=2560,
            )
        else:
            # 本地 Qwen3-0.6B，通过自定义 wrapper
            embeddings = _LocalQwen3Embeddings()

        t0 = time.time()
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
        build_sec = time.time() - t0

        return {
            "build_seconds": round(build_sec, 1),
            "backend": self.embedding_backend,
            "documents": len(texts),
        }

    def search(self, query: str) -> tuple[list[Hit], float]:
        assert self.vectorstore is not None
        t0 = time.time()
        docs = self.vectorstore.similarity_search_with_score(
            query, k=self.top_k,
        )
        elapsed = time.time() - t0
        hits = [
            Hit(text=doc.page_content, score=float(score))
            for doc, score in docs
        ]
        return hits, elapsed


class _LocalQwen3Embeddings:
    """LangChain Embeddings 接口包装本地 Qwen3-0.6B"""

    def __init__(self):
        from memento.index.vector_index import VectorIndex
        self._vi = VectorIndex(
            model_name=str(ROOT / "models" / "Qwen3-Embedding-0.6B"),
        )
        # 触发模型加载
        self._vi._ensure_fitted()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._vi.encode(texts, mode="document")
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._vi.encode([text], mode="query")[0]
        return vec.tolist()


# ──────────────────── 运行对比 ────────────────────

def run_adapter(adapter, pairs, queries) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  {adapter.name}")
    print(f"{'='*60}", flush=True)

    try:
        build_info = adapter.build(pairs)
    except Exception as exc:
        print(f"[SKIP] {adapter.name}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return {"available": False, "error": str(exc), "queries": {}}

    print(f"build: {json.dumps(build_info, ensure_ascii=False)}", flush=True)

    query_results = {}
    total_ms = 0.0
    for query in queries:
        try:
            hits, elapsed = adapter.search(query)
            ms = elapsed * 1000
            total_ms += ms
            query_results[query] = {
                "time_ms": round(ms, 1),
                "hits": [
                    {"text": hit.text, "score": hit.score}
                    for hit in hits
                ],
            }
            print(f"\nQ: {query} [{ms:.0f}ms]", flush=True)
            for rank, hit in enumerate(hits, 1):
                score = "-" if hit.score is None else f"{hit.score:.4f}"
                print(f"  {rank}. [{score}] {truncate(hit.text, 100)}", flush=True)
        except Exception as exc:
            query_results[query] = {"error": str(exc), "hits": []}
            print(f"\nQ: {query} -> ERROR: {exc}", flush=True)

    avg_ms = total_ms / len(queries) if queries else 0
    print(f"\n平均查询耗时: {avg_ms:.0f}ms", flush=True)

    return {
        "available": True,
        "build": build_info,
        "queries": query_results,
        "avg_query_ms": round(avg_ms, 1),
    }


def write_comparison(results: dict[str, Any], run_id: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / f"{run_id}_langchain.md"
    json_path = RESULTS_DIR / f"{run_id}_langchain.json"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Memory Benchmark: Memento vs LangChain (标准 RAG)",
        "",
        f"- Run: `{run_id}`",
        f"- Data: {results['data']['count']} Q&A pairs",
        f"- Queries: {len(results['queries'])}",
        "",
        "## 对比说明",
        "",
        "| 系统 | Embedding | 检索方式 | 概念图 |",
        "|------|-----------|----------|--------|",
        "| Memento | Qwen3-0.6B (本地) | RAG + 概念图扩散 | Yes |",
        "| LangChain-SF | Qwen3-4B (SiliconFlow) | 纯 FAISS top-k | No |",
        "| LangChain-Local | Qwen3-0.6B (本地) | 纯 FAISS top-k | No |",
        "",
    ]

    # 汇总表
    lines.extend([
        "## 汇总",
        "",
        "| 系统 | 构建耗时 | 平均查询ms |",
        "|------|----------|-----------|",
    ])
    for sys_name, sys_result in results["systems"].items():
        if not sys_result.get("available"):
            lines.append(f"| {sys_name} | SKIP ({sys_result.get('error', '')[:30]}) | - |")
            continue
        build = sys_result.get("build", {})
        build_time = build.get("index_seconds", 0) + build.get("concept_seconds", 0) + build.get("build_seconds", 0)
        avg_ms = sys_result.get("avg_query_ms", "-")
        lines.append(f"| {sys_name} | {build_time:.1f}s | {avg_ms}ms |")
    lines.append("")

    # 逐 query 对比
    for query in results["queries"]:
        lines.extend([f"## Q: {query}", ""])
        for sys_name, sys_result in results["systems"].items():
            if not sys_result.get("available"):
                continue
            qr = sys_result["queries"].get(query, {})
            if "error" in qr:
                lines.append(f"### {sys_name}: ERROR - {qr['error']}")
                continue
            lines.append(f"### {sys_name} ({qr.get('time_ms', '?')}ms)")
            lines.append("")
            for rank, hit in enumerate(qr.get("hits", []), 1):
                score = hit.get("score")
                score_text = "-" if score is None else f"{score:.4f}"
                lines.append(f"{rank}. `score={score_text}` {truncate(hit.get('text', ''), 160)}")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))

    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"数据: {len(pairs)} 条 Q&A", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    top_k = 5

    adapters = [
        MementoAdapter(top_k),
        LangChainFAISSAdapter(top_k, embedding_backend="siliconflow"),
        LangChainFAISSAdapter(top_k, embedding_backend="local"),
    ]

    all_results = {
        "run_id": run_id,
        "data": {"count": len(pairs)},
        "queries": QUERIES,
        "systems": {},
    }

    for adapter in adapters:
        result = run_adapter(adapter, pairs, QUERIES)
        all_results["systems"][adapter.name] = result

    md_path = write_comparison(all_results, run_id)
    print(f"\n{'='*60}")
    print(f"结果已写入: {md_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
