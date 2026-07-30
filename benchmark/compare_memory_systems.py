"""
统一记忆命中评测主线。

当前只输出主干观察结果，不计算 Hit@1/MRR 等量化指标。
Letta 固定走 Docker HTTP API，不 import 本地 letta/letta_client。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"
LETTA_BASE_URL = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
LETTA_PROVIDER_NAME = "opencode-deepseek-v4-flash"

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


@dataclass
class Hit:
    text: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


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


def normalize_systems(value: str) -> list[str]:
    systems = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"memento", "mem0", "letta"}
    unknown = [item for item in systems if item not in allowed]
    if unknown:
        raise ValueError(f"未知系统: {unknown}; 可选: {sorted(allowed)}")
    return systems


class MementoAdapter:
    name = "Memento"

    def __init__(self, top_k: int, use_concepts: bool,
                 embedding_model: str | None = None) -> None:
        self.top_k = top_k
        self.use_concepts = use_concepts
        self.embedding_model = embedding_model or str(ROOT / "models" / "Qwen3-Embedding-0.6B")
        self.engine = None

    def build(self, pairs: list[dict[str, str]]) -> dict[str, Any]:
        from memento.api import Memento

        self.engine = Memento(embedding_model=self.embedding_model, diffusion_hops=2)
        t0 = time.time()
        for index, pair in enumerate(pairs):
            self.engine.add_node(
                text=memory_text(pair),
                node_id=f"qa_{index:04d}",
                importance=0.5,
            )
        indexed = self.engine.build_index()
        index_seconds = time.time() - t0

        concept_info = None
        concept_seconds = 0.0
        if self.use_concepts:
            t1 = time.time()
            concept_info = self.engine.build_concept_graph(
                top_k=8,
                keyword_model="models/Qwen3-Embedding-0.6B",
                keyword_device=None,
                keyword_dtype="float16",
                keyword_cache_enabled=True,
                keyword_cache_dir="data/keyatten_cache",
                keyword_sim_threshold=0.45,
                keyword_temperature=0.06,
                keyword_top_neighbors=8,
                max_concepts=500,
                min_concept_energy=0.5,
            )
            concept_seconds = time.time() - t1

        return {
            "indexed": indexed,
            "index_seconds": index_seconds,
            "concept_seconds": concept_seconds,
            "concept_info": concept_info,
        }

    def search(self, query: str) -> tuple[list[Hit], float]:
        assert self.engine is not None
        t0 = time.time()
        if self.use_concepts:
            raw_hits = self.engine.query_with_concepts(
                query,
                k=self.top_k,
                seed_k=max(20, self.top_k),
                concept_k=10,
                concept_hops=2,
                concept_weight=0.45,
            )
        else:
            raw_hits = self.engine.query(query, k=self.top_k, seed_k=max(20, self.top_k))
        elapsed = time.time() - t0
        hits = [
            Hit(
                text=item["text"],
                score=float(item.get("score", 0.0)),
                metadata={
                    key: item[key]
                    for key in ("rag_score", "concept_score", "id")
                    if key in item
                },
            )
            for item in raw_hits
        ]
        return hits, elapsed


class Mem0Adapter:
    name = "Mem0"

    def __init__(self, top_k: int, run_id: str) -> None:
        self.top_k = top_k
        self.run_id = run_id
        self.memory = None
        self.user_id = f"benchmark_{run_id}"

    def build(self, pairs: list[dict[str, str]]) -> dict[str, Any]:
        from mem0 import Memory

        os.environ["MEM0_TELEMETRY"] = "false"
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "deepseek-v4-flash",
                    "api_key": require_env("OPENCODE_API_KEY"),
                    "openai_base_url": require_env("OPENCODE_API_BASE"),
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "Qwen/Qwen3-Embedding-4B",
                    "api_key": require_env("SILICONFLOW_API_KEY"),
                    "openai_base_url": require_env("SILICONFLOW_API_BASE"),
                    "embedding_dims": 2560,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"memento_benchmark_{self.run_id}",
                    "embedding_model_dims": 2560,
                    "on_disk": True,
                    "path": str(ROOT / "benchmark" / "_mem0_qdrant").replace("\\", "/"),
                },
            },
        }
        t0 = time.time()
        self.memory = Memory.from_config(config)
        init_seconds = time.time() - t0

        t1 = time.time()
        stored = 0
        errors = 0
        for index, pair in enumerate(pairs):
            try:
                self.memory.add(
                    memory_text(pair),
                    user_id=self.user_id,
                    infer=False,
                    metadata={"idx": index, "benchmark_run": self.run_id},
                )
                stored += 1
            except Exception:
                errors += 1
        return {
            "init_seconds": init_seconds,
            "build_seconds": time.time() - t1,
            "stored": stored,
            "errors": errors,
        }

    def search(self, query: str) -> tuple[list[Hit], float]:
        assert self.memory is not None
        t0 = time.time()
        raw = self.memory.search(
            query,
            filters={"user_id": self.user_id},
            top_k=self.top_k,
        )
        elapsed = time.time() - t0
        items = raw.get("results", []) if isinstance(raw, dict) else raw
        hits = [
            Hit(
                text=item.get("memory", "") or item.get("text", ""),
                score=float(item.get("score", 0.0)),
                metadata=item.get("metadata") or {},
            )
            for item in items
        ]
        return hits, elapsed


class LettaHttpAdapter:
    name = "Letta"

    def __init__(self, top_k: int, run_id: str, base_url: str = LETTA_BASE_URL) -> None:
        self.top_k = top_k
        self.run_id = run_id
        self.base_url = base_url.rstrip("/")
        self.agent_id: str | None = None
        self.tag = f"memento_benchmark_{run_id}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = requests.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        if not response.text.strip():
            return None
        return response.json()

    def _health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v1/health", timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _ensure_provider(self) -> dict[str, Any]:
        providers = self._request(
            "GET",
            "v1/providers/",
            params={"name": LETTA_PROVIDER_NAME},
        )
        payload = {
            "api_key": require_env("OPENCODE_API_KEY"),
            "base_url": require_env("OPENCODE_API_BASE"),
        }
        if providers:
            provider_id = providers[0]["id"]
            return self._request("PATCH", f"v1/providers/{provider_id}", json=payload)
        return self._request(
            "POST",
            "v1/providers/",
            json={
                "name": LETTA_PROVIDER_NAME,
                "provider_type": "openai",
                **payload,
            },
        )

    def build(self, pairs: list[dict[str, str]]) -> dict[str, Any]:
        if not self._health():
            raise RuntimeError(
                f"Letta Docker server 不可用: {self.base_url}. "
                "先运行 docker compose -f scripts/test/letta-compose.yaml up -d"
            )

        provider = self._ensure_provider()
        agent = self._request(
            "POST",
            "v1/agents/",
            json={
                "name": f"memento_benchmark_{self.run_id}",
                "llm_config": {
                    "model": "deepseek-v4-flash",
                    "model_endpoint_type": "openai",
                    "model_endpoint": require_env("OPENCODE_API_BASE"),
                    "provider_name": LETTA_PROVIDER_NAME,
                    "provider_category": "byok",
                    "context_window": 30000,
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
                "embedding_config": {
                    "embedding_endpoint_type": "openai",
                    "embedding_endpoint": require_env("SILICONFLOW_API_BASE"),
                    "embedding_model": "Qwen/Qwen3-Embedding-4B",
                    "embedding_dim": 2560,
                    "embedding_chunk_size": 300,
                },
                "include_base_tools": False,
                "include_default_source": False,
                "memory_blocks": [],
                "tags": [self.tag],
            },
        )
        self.agent_id = agent["id"]

        t0 = time.time()
        stored = 0
        for index, pair in enumerate(pairs):
            self._request(
                "POST",
                f"v1/agents/{self.agent_id}/archival-memory",
                json={
                    "text": memory_text(pair),
                    "tags": [self.tag, f"idx:{index}"],
                },
            )
            stored += 1
        return {
            "provider_id": provider["id"],
            "agent_id": self.agent_id,
            "build_seconds": time.time() - t0,
            "stored": stored,
        }

    def search(self, query: str) -> tuple[list[Hit], float]:
        assert self.agent_id is not None
        t0 = time.time()
        raw = self._request(
            "GET",
            f"v1/agents/{self.agent_id}/archival-memory/search",
            params={"query": query, "tags": self.tag, "top_k": self.top_k},
        )
        elapsed = time.time() - t0
        results = raw.get("results", []) if isinstance(raw, dict) else []
        hits = [
            Hit(
                text=item.get("content", ""),
                score=item.get("score"),
                metadata={key: item.get(key) for key in ("id", "tags") if key in item},
            )
            for item in results
        ]
        return hits, elapsed


def run_adapter(adapter, pairs: list[dict[str, str]], queries: list[str]) -> dict[str, Any]:
    print(f"\n=== {adapter.name} ===", flush=True)
    try:
        build_info = adapter.build(pairs)
    except Exception as exc:
        print(f"[SKIP] {adapter.name}: {exc}", flush=True)
        return {"available": False, "error": str(exc), "queries": {}}

    print(f"build: {build_info}", flush=True)
    query_results: dict[str, Any] = {}
    for query in queries:
        try:
            hits, elapsed = adapter.search(query)
            query_results[query] = {
                "time_ms": round(elapsed * 1000, 1),
                "hits": [
                    {
                        "text": hit.text,
                        "score": hit.score,
                        "metadata": hit.metadata or {},
                    }
                    for hit in hits
                ],
            }
            print(f"\nQ: {query} [{elapsed * 1000:.0f}ms]", flush=True)
            for rank, hit in enumerate(hits, 1):
                score = "-" if hit.score is None else f"{hit.score:.4f}"
                print(f"  {rank}. [{score}] {truncate(hit.text, 100)}", flush=True)
        except Exception as exc:
            query_results[query] = {"error": str(exc), "hits": []}
            print(f"\nQ: {query} -> ERROR: {exc}", flush=True)

    return {
        "available": True,
        "build": build_info,
        "queries": query_results,
    }


def write_outputs(result: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{run_id}.json"
    md_path = RESULTS_DIR / f"{run_id}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Memory Benchmark",
        "",
        f"- Run: `{run_id}`",
        f"- Data: `{result['data']['path']}`, {result['data']['count']} Q&A",
        f"- Systems: {', '.join(result['systems'].keys())}",
        "- Metrics: 主干观察，不做量化打分",
        "",
    ]
    for system_name, system_result in result["systems"].items():
        lines.extend([f"## {system_name}", ""])
        if not system_result.get("available"):
            lines.extend([f"- SKIP: {system_result.get('error')}", ""])
            continue
        lines.extend(["### Build", "", "```json"])
        lines.append(json.dumps(system_result.get("build", {}), ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
        for query, query_result in system_result["queries"].items():
            lines.extend([f"### {query}", ""])
            if "error" in query_result:
                lines.extend([f"- ERROR: {query_result['error']}", ""])
                continue
            lines.append(f"- Time: {query_result.get('time_ms')}ms")
            lines.append("")
            for rank, hit in enumerate(query_result.get("hits", []), 1):
                score = hit.get("score")
                score_text = "-" if score is None else f"{score:.4f}"
                lines.append(f"{rank}. `score={score_text}` {truncate(hit.get('text', ''), 180)}")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="memento,letta")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-concepts", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))

    systems = normalize_systems(args.systems)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt", limit=args.limit)
    print(f"data: {len(pairs)} Q&A", flush=True)
    print(f"systems: {systems}", flush=True)

    adapters = []
    if "memento" in systems:
        adapters.append(MementoAdapter(args.top_k, use_concepts=not args.no_concepts))
    if "mem0" in systems:
        adapters.append(Mem0Adapter(args.top_k, run_id=run_id))
    if "letta" in systems:
        adapters.append(LettaHttpAdapter(args.top_k, run_id=run_id))

    result = {
        "run_id": run_id,
        "data": {
            "path": "data/testtxt.txt",
            "count": len(pairs),
            "limit": args.limit,
        },
        "queries": QUERIES,
        "systems": {},
    }
    for adapter in adapters:
        result["systems"][adapter.name] = run_adapter(adapter, pairs, QUERIES)

    md_path, json_path = write_outputs(result, run_id)
    print(f"\nmarkdown: {md_path}", flush=True)
    print(f"json: {json_path}", flush=True)


if __name__ == "__main__":
    main()
