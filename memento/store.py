"""
Memento 存储助手 — CLI / HTTP / MCP 三端共用

负责:
  - 解析存储目录（--store / MEMENTO_STORE 环境变量 / 默认 data/memento_store）
  - load_or_create(): 从磁盘加载已有记忆系统，不存在则返回空实例
  - save(): 落盘 + 写 store_meta.json 记录 embedding_model，下次加载自动对齐
  - node_to_dict(): 把 Node 序列化成 JSON 友好结构（去掉 vector）

这样三端不用各自处理“首次运行没有 store”“embedding model 不一致”等问题。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from memento.api import Memento
from memento.models import Node

DEFAULT_STORE = "data/memento_store"
DEFAULT_EMBEDDING = "tfidf-svd"

_META_FILE = "store_meta.json"


def resolve_store(store: Optional[str] = None) -> str:
    """存储目录优先级: 显式传入 > MEMENTO_STORE 环境变量 > 默认值。"""
    if store:
        return store
    return os.environ.get("MEMENTO_STORE", DEFAULT_STORE)


def resolve_embedding_model(
    embedding_model: Optional[str] = None,
    store: Optional[str] = None,
) -> str:
    """embedding 模型优先级: 显式传入 > 环境变量 > store_meta 记录 > 默认 tfidf-svd。"""
    if embedding_model:
        return embedding_model
    env_model = os.environ.get("MEMENTO_EMBEDDING_MODEL")
    if env_model:
        return env_model
    saved = _read_store_meta(store).get("embedding_model")
    if saved:
        return saved
    return DEFAULT_EMBEDDING


def _read_store_meta(store: Optional[str] = None) -> dict:
    path = Path(resolve_store(store)) / _META_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def has_store(store: Optional[str] = None) -> bool:
    """判断 store 是否已有持久化数据（nodes.json 存在即视为已初始化）。"""
    return (Path(resolve_store(store)) / "nodes.json").exists()


def load_or_create(
    store: Optional[str] = None,
    embedding_model: Optional[str] = None,
    **memento_kwargs,
) -> Memento:
    """加载已有 store；若不存在则返回空 Memento 实例。

    embedding_model 不显式传入时，自动读取 store_meta / 环境变量 / 默认值，
    保证 save→load 循环里向量后端一致，避免 FAISS 维度不匹配。
    """
    path = Path(resolve_store(store))
    model = resolve_embedding_model(embedding_model, store)

    mem = Memento(embedding_model=model, **memento_kwargs)

    if (path / "nodes.json").exists():
        mem.load(str(path))
    return mem


def save(mem: Memento, store: Optional[str] = None) -> str:
    """落盘 Memento 实例，并记录 embedding_model 到 store_meta.json。"""
    path = Path(resolve_store(store))
    path.mkdir(parents=True, exist_ok=True)
    mem.save(str(path))

    meta = {
        "embedding_model": getattr(mem.vector_index, "_model_name", DEFAULT_EMBEDDING),
        "version": 1,
    }
    with open(path / _META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(path)


def node_to_dict(node: Node) -> dict:
    """Node → JSON 友好字典（排除 numpy vector）。"""
    return {
        "id": node.id,
        "text": node.text,
        "importance": round(float(node.importance), 4),
        "vitality": round(float(node.vitality), 4),
        "access_count": node.access_count,
        "edge_count": node.edge_count,
        "tags": list(node.tags),
        "source": node.source,
        "status": node.status,
        "created_at": node.created_at,
        "superseded_by": node.superseded_by,
        "fused_from": list(node.fused_from),
    }
