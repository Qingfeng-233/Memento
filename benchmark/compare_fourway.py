"""
Memento vs PureRAG vs mem0 三系统对比测试

- Memento: 本地 Qwen3-0.6B + 概念图扩散 + 上下文路由
- PureRAG: ChromaDB + Qwen3-4B embedding + 纯向量 cosine 检索
- mem0: Qdrant + Qwen3-4B embedding + LLM 事实提取
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
        pairs.append({"question": question.strip(), "answer": answer.strip()})
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def memory_text(pair: dict[str, str]) -> str:
    return f"用户问: {pair['question']}\n回答: {pair['answer']}"


def truncate(text: str, limit: int = 140) -> str:
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
            self.engine.add_node(text=memory_text(pair), node_id=f"qa_{idx:04d}", importance=0.5)
        self.engine.build_index()
        idx_sec = time.time() - t0

        t1 = time.time()
        info = self.engine.build_concept_graph(
            top_k=8, keyword_model=str(ROOT / "models" / "Qwen3-Embedding-0.6B"),
            keyword_dtype="float16", keyword_cache_dir="data/keyatten_cache",
            keyword_sim_threshold=0.45, keyword_temperature=0.06,
            keyword_top_neighbors=8, max_concepts=500, min_concept_energy=0.5,
        )
        cg_sec = time.time() - t1
        return {
            "index_s": round(idx_sec, 1), "concept_s": round(cg_sec, 1),
            "concepts": info["concepts"],
            "ctx_vectors": len(self.engine.concept_graph.context_vectors),
        }

    def search(self, query: str) -> tuple[list[Hit], float]:
        t0 = time.time()
        raw = self.engine.query_with_concepts(
            query, k=self.top_k, seed_k=20, concept_k=10,
            concept_hops=2, concept_weight=0.45,
        )
        elapsed = time.time() - t0
        hits = [Hit(text=r["text"], score=float(r.get("score", 0))) for r in raw]
        return hits, elapsed


# ──────────────────── PureRAG 适配器 ────────────────────

class PureRAGAdapter:
    """纯向量 RAG 基线：ChromaDB + Qwen3-Embedding-4B + cosine distance"""
    name = "PureRAG"

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.collection = None
        self._client = None

    def build(self, pairs: list[dict[str, str]]) -> dict[str, Any]:
        import shutil
        import chromadb
        from openai import OpenAI

        api_key = require_env("SILICONFLOW_API_KEY")
        api_base = require_env("SILICONFLOW_API_BASE")
        self._client = OpenAI(api_key=api_key, base_url=api_base)

        # 清理旧数据
        db_path = str(ROOT / "benchmark" / "_purerag_db")
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)

        chroma = chromadb.PersistentClient(path=db_path)
        self.collection = chroma.create_collection(
            name="benchmark",
            metadata={"hnsw:space": "cosine"},
        )

        texts = [memory_text(p) for p in pairs]
        ids = [f"mem_{i:04d}" for i in range(len(pairs))]

        t0 = time.time()
        # 分批 embedding
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            resp = self._client.embeddings.create(
                model="Qwen/Qwen3-Embedding-4B",
                input=texts[i:i + batch_size],
                dimensions=2560,
            )
            all_embeddings.extend([item.embedding for item in resp.data])

        self.collection.add(
            documents=texts, ids=ids, embeddings=all_embeddings,
        )
        build_sec = time.time() - t0

        count = self.collection.count()
        return {"build_s": round(build_sec, 1), "stored": count}

    def search(self, query: str) -> tuple[list[Hit], float]:
        # query embedding
        q_resp = self._client.embeddings.create(
            model="Qwen/Qwen3-Embedding-4B",
            input=[query],
            dimensions=2560,
        )
        q_emb = q_resp.data[0].embedding

        t0 = time.time()
        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
            include=["documents", "distances"],
        )
        elapsed = time.time() - t0

        hits = []
        if result:
            docs = result.get("documents", [[]])[0] if result.get("documents") else []
            dists = result.get("distances", [[]])[0] if result.get("distances") else []
            for doc, dist in zip(docs, dists):
                score = max(0.0, 1.0 - dist)  # cosine distance -> similarity
                hits.append(Hit(text=doc, score=score))
        return hits, elapsed


# ──────────────────── mem0 适配器 ────────────────────

class Mem0Adapter:
    """mem0: LLM 事实提取 + Qdrant 向量检索"""
    name = "mem0"

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.memory = None
        self.db_path = str(ROOT / "benchmark" / "_mem0_bench_compare")

    def build(self, pairs: list[dict[str, str]]) -> dict[str, Any]:
        import shutil
        from mem0 import Memory

        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path, ignore_errors=True)

        config = {
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "Qwen/Qwen3-Embedding-4B",
                    "api_key": require_env("SILICONFLOW_API_KEY"),
                    "openai_base_url": require_env("SILICONFLOW_API_BASE"),
                    "embedding_dims": 2560,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "deepseek-v4-flash",
                    "api_key": require_env("OPENCODE_API_KEY"),
                    "openai_base_url": require_env("OPENCODE_API_BASE"),
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_compare",
                    "embedding_model_dims": 2560,
                    "path": self.db_path,
                }
            },
            "version": "v1.1",
        }

        self.memory = Memory.from_config(config)

        t0 = time.time()
        errors = 0
        for i, pair in enumerate(pairs):
            text = memory_text(pair)
            try:
                self.memory.add(text, user_id="benchmark")
            except Exception:
                errors += 1
            if (i + 1) % 20 == 0:
                print(f"  mem0 build: {i+1}/{len(pairs)}", flush=True)

        build_sec = time.time() - t0

        all_memories = self.memory.get_all(user_id="benchmark")
        n_memories = len(all_memories.get("results", []))
        return {"build_s": round(build_sec, 1), "memories": n_memories, "errors": errors}

    def search(self, query: str) -> tuple[list[Hit], float]:
        t0 = time.time()
        results = self.memory.search(
            query, top_k=self.top_k, filters={"user_id": "benchmark"},
        )
        elapsed = time.time() - t0

        hits = []
        for r in results.get("results", []):
            hits.append(Hit(text=r.get("memory", ""), score=r.get("score")))
        return hits, elapsed


# ──────────────────── 运行对比 ────────────────────

def run_adapter(adapter, pairs, queries) -> dict[str, Any]:
    print(f"\n{'='*60}", flush=True)
    print(f"  {adapter.name}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        build_info = adapter.build(pairs)
    except Exception as exc:
        print(f"[SKIP] {adapter.name}: {exc}", flush=True)
        import traceback; traceback.print_exc()
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
                "hits": [{"text": h.text, "score": h.score} for h in hits],
            }
            print(f"\nQ: {query} [{ms:.0f}ms]", flush=True)
            for rank, hit in enumerate(hits, 1):
                s = "-" if hit.score is None else f"{hit.score:.4f}"
                print(f"  {rank}. [{s}] {truncate(hit.text, 100)}", flush=True)
        except Exception as exc:
            query_results[query] = {"error": str(exc), "hits": []}
            print(f"\nQ: {query} -> ERROR: {exc}", flush=True)

    avg_ms = total_ms / len(queries) if queries else 0
    print(f"\n平均查询: {avg_ms:.0f}ms", flush=True)

    return {
        "available": True,
        "build": build_info,
        "queries": query_results,
        "avg_query_ms": round(avg_ms, 1),
    }


def write_comparison(results: dict[str, Any], run_id: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / f"{run_id}_fourway.md"
    json_path = RESULTS_DIR / f"{run_id}_fourway.json"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 三系统对比: Memento vs PureRAG vs mem0",
        "",
        f"- Run: `{run_id}`",
        f"- Data: {results['data']['count']} Q&A",
        f"- Queries: {len(results['queries'])}",
        "",
        "## 系统简介",
        "",
        "| 系统 | 存储方式 | 检索方式 | 特点 |",
        "|------|----------|----------|------|",
        "| Memento | 向量 + 概念图 | RAG + 图扩散 + 上下文路由 | 双系统，关键词消歧 |",
        "| PureRAG | ChromaDB (Qwen3-4B) | 纯向量 cosine top-k | 最简 RAG 基线 |",
        "| mem0 | Qdrant + LLM 事实提取 | 向量检索 + 结构化事实 | LLM 提取关键信息 |",
        "",
    ]

    # 汇总表
    lines.extend(["## 性能汇总", "", "| 系统 | 构建耗时 | 平均查询ms | 状态 |", "|------|----------|-----------|------|"])
    for name, sr in results["systems"].items():
        if not sr.get("available"):
            lines.append(f"| {name} | - | - | SKIP: {sr.get('error', '')[:40]} |")
            continue
        b = sr.get("build", {})
        bt = b.get("index_s", 0) + b.get("concept_s", 0) + b.get("build_s", 0)
        lines.append(f"| {name} | {bt:.1f}s | {sr.get('avg_query_ms', '-')}ms | OK |")
    lines.append("")

    # 逐 query
    for query in results["queries"]:
        lines.extend([f"## Q: {query}", ""])
        for name, sr in results["systems"].items():
            if not sr.get("available"):
                continue
            qr = sr["queries"].get(query, {})
            if "error" in qr:
                lines.append(f"### {name}: ERROR - {qr['error'][:60]}")
                continue
            lines.append(f"### {name} ({qr.get('time_ms', '?')}ms)")
            lines.append("")
            for rank, hit in enumerate(qr.get("hits", []), 1):
                s = hit.get("score")
                st = "-" if s is None else f"{s:.4f}"
                lines.append(f"{rank}. `{st}` {truncate(hit.get('text', ''), 160)}")
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
        PureRAGAdapter(top_k),
        Mem0Adapter(top_k),
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
    print(f"结果: {md_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
