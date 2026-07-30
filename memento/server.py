"""
Memento HTTP 服务（FastAPI）

对外接口，常驻进程。启动时从磁盘加载记忆系统，写操作后自动落盘
（可用 --no-autosave 关闭，转而手动 POST /save）。

启动:
    python -m memento.server [--host 0.0.0.0] [--port 8765] [--store PATH]
    # 或通过 CLI:
    python -m memento.cli serve

端点:
    POST /add               存入记忆
    POST /import            批量导入 jsonl 行
    POST /build-index       构建向量索引
    POST /build-concept-graph   构建关键词概念图
    POST /build-keyword-edges   构建关键词重叠边
    POST /query             向量 + 扩散检索
    POST /query-concepts    概念图检索
    POST /query-rag         纯 RAG 检索
    POST /activate          激活节点（情境共现建边）
    POST /link              连接两个节点
    POST /link-concepts     连接两个关键词概念
    POST /mark-important    调整重要性
    POST /clock-step        推进一个时钟步
    POST /sleep             触发睡眠巩固
    POST /save              手动落盘
    POST /load              从磁盘重新加载（丢弃内存改动）
    GET  /node/{node_id}    取节点
    GET  /stats             系统状态
    GET  /                  健康检查 + 端点列表
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memento import store as store_mod
from memento.api import Memento


# ─── 请求模型 ──────────────────────────────────────────────


class AddRequest(BaseModel):
    text: str
    id: Optional[str] = None
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    source: str = "api"
    created_at: Optional[str] = None
    build: bool = False


class ImportRequest(BaseModel):
    items: list[dict]


class BuildConceptGraphRequest(BaseModel):
    top_k: int = 8
    keyword_method: str = "keyatten"
    max_concepts: int = 300
    min_concept_energy: float = 0.5
    keyword_sim_threshold: float = 0.65
    keyword_temperature: float = 0.08
    keyword_top_neighbors: int = 5
    dedup_concepts: bool = False
    dedup_threshold: float = 0.90
    keyword_model: str = "models/Qwen3-Embedding-0.6B"
    keyword_device: Optional[str] = None
    keyword_dtype: Optional[str] = "float16"
    keyword_cache_enabled: bool = True
    keyword_cache_dir: str = "data/keyatten_cache"


class BuildKeywordEdgesRequest(BaseModel):
    top_k: int = 5
    min_overlap: int = 1
    max_node_freq: int = 20
    weight_per_keyword: float = 0.15
    max_weight: float = 0.6
    semantic_filter: bool = False
    min_cos_sim: float = 0.30
    keyword_model: str = "models/Qwen3-Embedding-0.6B"
    keyword_device: Optional[str] = None
    keyword_dtype: Optional[str] = "float16"
    keyword_cache_enabled: bool = True
    keyword_cache_dir: str = "data/keyatten_cache"


class QueryRequest(BaseModel):
    text: str
    k: int = 10
    seed_k: int = 20


class QueryConceptsRequest(BaseModel):
    text: str
    k: int = 10
    seed_k: int = 20
    concept_k: int = 8
    concept_hops: int = 2
    concept_weight: float = 0.35
    debug: bool = False


class QueryRagRequest(BaseModel):
    text: str
    k: int = 10


class ActivateRequest(BaseModel):
    node_ids: list[str]


class LinkRequest(BaseModel):
    node_a: str
    node_b: str
    weight: float = 0.8


class LinkConceptsRequest(BaseModel):
    source: str
    target: str
    weight: float = 0.8


class MarkImportantRequest(BaseModel):
    node_id: str
    importance: float = 1.0


# ─── 应用状态 ──────────────────────────────────────────────


class AppState:
    mem: Memento
    store: Optional[str]
    autosave: bool


def create_app(
    store: Optional[str] = None,
    embedding_model: Optional[str] = None,
    no_autosave: bool = False,
) -> FastAPI:
    state = AppState()
    state.store = store
    state.autosave = not no_autosave
    state.mem = store_mod.load_or_create(store, embedding_model)

    app = FastAPI(
        title="Memento HTTP API",
        description="双系统联想记忆引擎 — 对外 HTTP 接口",
        version="0.1.0",
    )
    app.state.memento = state

    def _autosave():
        if state.autosave:
            store_mod.save(state.mem, state.store)

    @app.get("/")
    def root():
        return {
            "service": "memento",
            "version": "0.1.0",
            "store": store_mod.resolve_store(state.store),
            "autosave": state.autosave,
            "stats": state.mem.stats,
            "endpoints": [
                "POST /add",
                "POST /import",
                "POST /build-index",
                "POST /build-concept-graph",
                "POST /build-keyword-edges",
                "POST /query",
                "POST /query-concepts",
                "POST /query-rag",
                "POST /activate",
                "POST /link",
                "POST /link-concepts",
                "POST /mark-important",
                "POST /clock-step",
                "POST /sleep",
                "POST /save",
                "POST /load",
                "GET /node/{node_id}",
                "GET /stats",
            ],
        }

    @app.get("/stats")
    def stats():
        return state.mem.stats

    @app.get("/node/{node_id}")
    def get_node(node_id: str):
        node = state.mem.get_node(node_id)
        if node is None:
            raise HTTPException(404, f"node {node_id} not found")
        return store_mod.node_to_dict(node)

    @app.post("/add")
    def add(req: AddRequest):
        node_id = state.mem.add_node(
            text=req.text,
            node_id=req.id,
            importance=req.importance,
            tags=req.tags,
            source=req.source,
            created_at=req.created_at,
        )
        index_built = False
        added = None
        if req.build:
            added = state.mem.build_index()
            index_built = True
        _autosave()
        return {"id": node_id, "index_built": index_built, "added": added}

    @app.post("/import")
    def import_items(req: ImportRequest):
        count = 0
        for m in req.items:
            state.mem.add_node(
                text=m["text"],
                node_id=m.get("id"),
                importance=m.get("importance", 0.5),
                tags=m.get("tags", []),
                source=m.get("source", "api"),
                created_at=m.get("created_at"),
            )
            count += 1
        _autosave()
        return {"imported": count, "hint": "调用 /build-index 后才可查询"}

    @app.post("/build-index")
    def build_index():
        count = state.mem.build_index()
        _autosave()
        return {"added": count}

    @app.post("/build-concept-graph")
    def build_concept_graph(req: BuildConceptGraphRequest):
        info = state.mem.build_concept_graph(
            top_k=req.top_k,
            keyword_method=req.keyword_method,
            max_concepts=req.max_concepts,
            min_concept_energy=req.min_concept_energy,
            keyword_sim_threshold=req.keyword_sim_threshold,
            keyword_temperature=req.keyword_temperature,
            keyword_top_neighbors=req.keyword_top_neighbors,
            dedup_concepts=req.dedup_concepts,
            dedup_threshold=req.dedup_threshold,
            keyword_model=req.keyword_model,
            keyword_device=req.keyword_device,
            keyword_dtype=req.keyword_dtype,
            keyword_cache_enabled=req.keyword_cache_enabled,
            keyword_cache_dir=req.keyword_cache_dir,
        )
        _autosave()
        return info

    @app.post("/build-keyword-edges")
    def build_keyword_edges(req: BuildKeywordEdgesRequest):
        info = state.mem.build_keyword_edges(
            top_k=req.top_k,
            min_overlap=req.min_overlap,
            max_node_freq=req.max_node_freq,
            weight_per_keyword=req.weight_per_keyword,
            max_weight=req.max_weight,
            semantic_filter=req.semantic_filter,
            min_cos_sim=req.min_cos_sim,
            keyword_model=req.keyword_model,
            keyword_device=req.keyword_device,
            keyword_dtype=req.keyword_dtype,
            keyword_cache_enabled=req.keyword_cache_enabled,
            keyword_cache_dir=req.keyword_cache_dir,
        )
        _autosave()
        return info

    @app.post("/query")
    def query(req: QueryRequest):
        return state.mem.query(req.text, k=req.k, seed_k=req.seed_k)

    @app.post("/query-concepts")
    def query_concepts(req: QueryConceptsRequest):
        return state.mem.query_with_concepts(
            req.text,
            k=req.k,
            seed_k=req.seed_k,
            concept_k=req.concept_k,
            concept_hops=req.concept_hops,
            concept_weight=req.concept_weight,
            debug=req.debug,
        )

    @app.post("/query-rag")
    def query_rag(req: QueryRagRequest):
        return state.mem.query_rag_only(req.text, k=req.k)

    @app.post("/activate")
    def activate(req: ActivateRequest):
        state.mem.activate(req.node_ids)
        _autosave()
        return {"ok": True, "activated": req.node_ids}

    @app.post("/link")
    def link(req: LinkRequest):
        state.mem.link(req.node_a, req.node_b, weight=req.weight)
        _autosave()
        return {"ok": True}

    @app.post("/link-concepts")
    def link_concepts(req: LinkConceptsRequest):
        state.mem.link_concepts(req.source, req.target, weight=req.weight)
        _autosave()
        return {"ok": True}

    @app.post("/mark-important")
    def mark_important(req: MarkImportantRequest):
        state.mem.mark_important(req.node_id, importance=req.importance)
        _autosave()
        return {"ok": True}

    @app.post("/clock-step")
    def clock_step():
        state.mem.clock_step()
        _autosave()
        return {"ok": True, "clock_step": state.mem._clock_step}

    @app.post("/sleep")
    def sleep():
        report = state.mem.trigger_sleep()
        _autosave()
        return asdict(report)

    @app.post("/save")
    def save():
        path = store_mod.save(state.mem, state.store)
        return {"ok": True, "path": path}

    @app.post("/load")
    def load():
        state.mem = store_mod.load_or_create(state.store)
        return {"ok": True, "stats": state.mem.stats}

    return app


def run_server(
    store: Optional[str] = None,
    embedding_model: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    no_autosave: bool = False,
):
    """CLI 入口: 构造 app 并用 uvicorn 启动。"""
    import uvicorn

    app = create_app(
        store=store,
        embedding_model=embedding_model,
        no_autosave=no_autosave,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m memento.server",
        description="Memento HTTP 服务",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--store", default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--no-autosave", action="store_true")
    args = p.parse_args()
    run_server(
        store=args.store,
        embedding_model=args.embedding_model,
        host=args.host,
        port=args.port,
        no_autosave=args.no_autosave,
    )
