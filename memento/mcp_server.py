"""
Memento MCP 服务（Model Context Protocol）

把 Memento 完整 API 暴露为 MCP 工具，供 LLM 客户端（Claude Desktop、
OpenCode 等）通过 stdio 调用。常驻进程，启动时从磁盘加载，写操作后自动落盘。

启动:
    python -m memento.mcp_server [--store PATH] [--no-autosave]
    # 或通过 CLI:
    python -m memento.cli mcp

Claude Desktop 配置示例 (claude_desktop_config.json):
{
  "mcpServers": {
    "memento": {
      "command": "python",
      "args": ["-m", "memento.mcp_server", "--store", "D:/工作区/项目/Memento/data/memento_store"]
    }
  }
}

暴露的工具（完整 API）:
    add_memory / add_memory_live / import_memories
    build_index / build_concept_graph / build_keyword_edges
    query / query_with_concepts / query_rag_only
    activate / link / link_concepts / mark_important
    get_node / get_node_keywords / get_keyword_surprisal
    clock_step / trigger_sleep / stats / save / load
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from memento import store as store_mod
from memento.api import Memento


# ─── 全局状态 ──────────────────────────────────────────────


class _State:
    mem: Optional[Memento] = None
    store: Optional[str] = None
    autosave: bool = True

    def ensure(self) -> Memento:
        if self.mem is None:
            self.mem = store_mod.load_or_create(self.store)
        return self.mem

    def save_if_enabled(self):
        if self.autosave and self.mem is not None:
            store_mod.save(self.mem, self.store)


_state = _State()
mcp = FastMCP("memento")


def _json(obj) -> str:
    """统一 JSON 字符串返回，保证跨 MCP 客户端兼容。"""
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ─── 写入工具 ──────────────────────────────────────────────


@mcp.tool()
def add_memory(
    text: str,
    id: Optional[str] = None,
    importance: float = 0.5,
    tags: Optional[list[str]] = None,
    source: str = "mcp",
    created_at: Optional[str] = None,
    build: bool = False,
) -> str:
    """存入一条记忆。build=True 时同时构建向量索引（之后才可查询）。

    Args:
        text: 记忆文本（通常是一段 Q/A 或笔记）
        id: 节点 id，留空自动生成
        importance: 重要性 0~1
        tags: 标签列表
        source: 来源标记
        created_at: 创建时间字符串
        build: 是否立即构建向量索引
    Returns:
        JSON: {"id": ..., "index_built": ..., "added": ...}
    """
    mem = _state.ensure()
    node_id = mem.add_node(
        text=text,
        node_id=id,
        importance=importance,
        tags=tags or [],
        source=source,
        created_at=created_at,
    )
    added = None
    if build:
        added = mem.build_index()
    _state.save_if_enabled()
    return _json({"id": node_id, "index_built": build, "added": added})


@mcp.tool()
def add_memory_live(
    text: str,
    id: Optional[str] = None,
    importance: float = 0.5,
    tags: Optional[list[str]] = None,
    source: str = "mcp",
    created_at: Optional[str] = None,
) -> str:
    """实时添加节点（索引已构建后可用，增量编码，无需 rebuild）。

    Returns:
        JSON: {"id": ...} 或错误信息
    """
    mem = _state.ensure()
    try:
        node_id = mem.add_node_live(
            text=text,
            node_id=id,
            importance=importance,
            tags=tags or [],
            source=source,
            created_at=created_at,
        )
    except RuntimeError as e:
        return _json({"error": str(e), "hint": "先调用 build_index 构建索引"})
    _state.save_if_enabled()
    return _json({"id": node_id})


@mcp.tool()
def import_memories(items: list[dict]) -> str:
    """批量导入记忆。每项 {"text", "id?", "importance?", "tags?", ...}。

    导入后需调用 build_index 才能查询。
    """
    mem = _state.ensure()
    count = 0
    for m in items:
        mem.add_node(
            text=m["text"],
            node_id=m.get("id"),
            importance=m.get("importance", 0.5),
            tags=m.get("tags", []),
            source=m.get("source", "mcp"),
            created_at=m.get("created_at"),
        )
        count += 1
    _state.save_if_enabled()
    return _json({"imported": count})


@mcp.tool()
def build_index() -> str:
    """构建向量索引（pending 节点编码并入 FAISS）。"""
    mem = _state.ensure()
    count = mem.build_index()
    _state.save_if_enabled()
    return _json({"added": count})


@mcp.tool()
def build_concept_graph(
    top_k: int = 8,
    keyword_method: str = "keyatten",
    max_concepts: int = 300,
    min_concept_energy: float = 0.5,
    keyword_sim_threshold: float = 0.65,
    keyword_temperature: float = 0.08,
    dedup_concepts: bool = False,
    dedup_threshold: float = 0.90,
    keyword_model: str = "models/Qwen3-Embedding-0.6B",
    keyword_device: Optional[str] = None,
    keyword_dtype: Optional[str] = "float16",
) -> str:
    """构建关键词副节点概念图。需先 build_index。

    关键词作为独立副节点，事件↔关键词按能量建边，关键词↔关键词按向量相似度建边。
    完成后可用 query_with_concepts 做概念图检索。

    Args:
        top_k: 每节点提取关键词数
        keyword_method: keyatten | statistical
        max_concepts: 概念数上限
        min_concept_energy: 概念初始能量下限（过滤泛词）
        keyword_sim_threshold: 关键词间建边的余弦阈值
        keyword_temperature: 边权指数化温度
        dedup_concepts: 合并高余弦近义锚点
        dedup_threshold: 去重余弦阈值
        keyword_model: keyatten 模型路径
        keyword_device: cuda | cpu（默认跟随主索引）
        keyword_dtype: 模型精度
    Returns:
        JSON: 概念图构建信息
    """
    mem = _state.ensure()
    info = mem.build_concept_graph(
        top_k=top_k,
        keyword_method=keyword_method,
        max_concepts=max_concepts,
        min_concept_energy=min_concept_energy,
        keyword_sim_threshold=keyword_sim_threshold,
        keyword_temperature=keyword_temperature,
        dedup_concepts=dedup_concepts,
        dedup_threshold=dedup_threshold,
        keyword_model=keyword_model,
        keyword_device=keyword_device,
        keyword_dtype=keyword_dtype,
    )
    _state.save_if_enabled()
    return _json(info)


@mcp.tool()
def build_keyword_edges(
    top_k: int = 5,
    min_overlap: int = 1,
    max_node_freq: int = 20,
    semantic_filter: bool = False,
    min_cos_sim: float = 0.30,
    keyword_model: str = "models/Qwen3-Embedding-0.6B",
    keyword_device: Optional[str] = None,
    keyword_dtype: Optional[str] = "float16",
) -> str:
    """基于关键词重叠建边。需先 build_index。

    Args:
        top_k: 每节点关键词数
        min_overlap: 最少共享关键词数才建边
        max_node_freq: 关键词出现在超过此数量节点中则跳过
        semantic_filter: 启用语义交叉过滤
        min_cos_sim: 语义过滤余弦阈值
    Returns:
        JSON: {"edges_added": ..., "edges_rejected": ..., ...}
    """
    mem = _state.ensure()
    info = mem.build_keyword_edges(
        top_k=top_k,
        min_overlap=min_overlap,
        max_node_freq=max_node_freq,
        semantic_filter=semantic_filter,
        min_cos_sim=min_cos_sim,
        keyword_model=keyword_model,
        keyword_device=keyword_device,
        keyword_dtype=keyword_dtype,
    )
    _state.save_if_enabled()
    return _json(info)


# ─── 检索工具 ──────────────────────────────────────────────


@mcp.tool()
def query(text: str, k: int = 10, seed_k: int = 20) -> str:
    """向量 RAG + 图扩散联想检索。

    Args:
        text: 查询文本
        k: 返回结果数
        seed_k: 向量召回种子数（扩散起点）
    Returns:
        JSON: 结果列表 [{id, text, score, importance, vitality, edges, tags}, ...]
    """
    mem = _state.ensure()
    return _json(mem.query(text, k=k, seed_k=seed_k))


@mcp.tool()
def query_with_concepts(
    text: str,
    k: int = 10,
    seed_k: int = 20,
    concept_k: int = 8,
    concept_hops: int = 2,
    concept_weight: float = 0.35,
    debug: bool = False,
) -> str:
    """概念图检索：向量 RAG + 关键词副节点扩散。需先 build_concept_graph。

    比纯 query 更能把字面不同但概念相近的记忆拉入候选。

    Args:
        text: 查询文本
        k: 返回结果数
        seed_k: 向量召回种子数
        concept_k: 激活的概念种子数
        concept_hops: 概念图扩散跳数
        concept_weight: 概念分权重（final = rag * (1 + concept_weight * concept_score)）
        debug: 返回 seed/activated concepts 调试信息
    Returns:
        JSON: debug=True 返回 {query, seed_concepts, activated_concepts, results}；
              否则返回结果列表
    """
    mem = _state.ensure()
    out = mem.query_with_concepts(
        text,
        k=k,
        seed_k=seed_k,
        concept_k=concept_k,
        concept_hops=concept_hops,
        concept_weight=concept_weight,
        debug=debug,
    )
    return _json(out)


@mcp.tool()
def query_rag_only(text: str, k: int = 10) -> str:
    """纯向量 RAG 检索（不扩散），用于和 query 对比。

    Returns:
        JSON: 结果列表
    """
    mem = _state.ensure()
    return _json(mem.query_rag_only(text, k=k))


# ─── 图操作工具 ────────────────────────────────────────────


@mcp.tool()
def activate(node_ids: list[str]) -> str:
    """激活一组节点 — 情境共现建边（窗口内节点两两加强边）。

    Args:
        node_ids: 要激活的节点 id 列表
    """
    mem = _state.ensure()
    mem.activate(node_ids)
    _state.save_if_enabled()
    return _json({"ok": True, "activated": node_ids})


@mcp.tool()
def link(node_a: str, node_b: str, weight: float = 0.8) -> str:
    """主动连接两个节点（手动建边）。

    Args:
        node_a: 节点 A id
        node_b: 节点 B id
        weight: 边权重 0~1
    """
    mem = _state.ensure()
    mem.link(node_a, node_b, weight=weight)
    _state.save_if_enabled()
    return _json({"ok": True})


@mcp.tool()
def link_concepts(source: str, target: str, weight: float = 0.8) -> str:
    """手工连接两个关键词概念（不存在时自动创建）。

    适合少量高置信的人工校正，不建议当主要数据来源。

    Args:
        source: 源概念文本
        target: 目标概念文本
        weight: 边权重 0~1
    """
    mem = _state.ensure()
    mem.link_concepts(source, target, weight=weight)
    _state.save_if_enabled()
    return _json({"ok": True})


@mcp.tool()
def mark_important(node_id: str, importance: float = 1.0) -> str:
    """调整节点重要性（0~1）。

    Args:
        node_id: 节点 id
        importance: 重要性 0~1
    """
    mem = _state.ensure()
    mem.mark_important(node_id, importance=importance)
    _state.save_if_enabled()
    return _json({"ok": True})


# ─── 读取工具 ──────────────────────────────────────────────


@mcp.tool()
def get_node(node_id: str) -> str:
    """按 id 取节点详情（文本、重要性、生命力、状态、标签等）。

    Returns:
        JSON: 节点字典 或 {"error": "not found"}
    """
    mem = _state.ensure()
    node = mem.get_node(node_id)
    if node is None:
        return _json({"error": "not found", "id": node_id})
    return _json(store_mod.node_to_dict(node))


@mcp.tool()
def get_node_keywords(node_id: str) -> str:
    """获取节点的关键词列表（需先 build_concept_graph 或 build_keyword_edges）。"""
    mem = _state.ensure()
    return _json(mem.get_node_keywords(node_id))


@mcp.tool()
def get_keyword_surprisal(node_id: str) -> str:
    """获取节点各关键词的惊奇度分数 1-cos（需先在 build_keyword_edges 中开启惊奇度）。"""
    mem = _state.ensure()
    return _json(mem.get_keyword_surprisal(node_id))


@mcp.tool()
def stats() -> str:
    """系统状态：节点数、边数、向量索引大小、时钟步等。"""
    mem = _state.ensure()
    return _json(mem.stats)


# ─── 生命周期工具 ──────────────────────────────────────────


@mcp.tool()
def clock_step() -> str:
    """推进一个时钟步（触发衰减）。"""
    mem = _state.ensure()
    mem.clock_step()
    _state.save_if_enabled()
    return _json({"ok": True, "clock_step": mem._clock_step})


@mcp.tool()
def trigger_sleep() -> str:
    """触发睡眠巩固周期（回放、漫游、探索、融合、聚类、遗忘修剪）。

    Returns:
        JSON: 睡眠报告（边强化/修剪、节点休眠、聚类等统计）
    """
    mem = _state.ensure()
    report = mem.trigger_sleep()
    _state.save_if_enabled()
    return _json(asdict(report))


@mcp.tool()
def save() -> str:
    """显式落盘到存储目录。"""
    mem = _state.ensure()
    path = store_mod.save(mem, _state.store)
    return _json({"ok": True, "path": path})


@mcp.tool()
def load() -> str:
    """从磁盘重新加载记忆系统（丢弃内存中未保存的改动）。"""
    _state.mem = store_mod.load_or_create(_state.store)
    return _json({"ok": True, "stats": _state.mem.stats})


# ─── 入口 ──────────────────────────────────────────────────


def run_mcp(
    store: Optional[str] = None,
    embedding_model: Optional[str] = None,
    no_autosave: bool = False,
):
    """CLI 入口: 配置全局状态并以 stdio transport 启动 MCP 服务。

    延迟加载 Memento 实例到首次工具调用，避免启动时阻塞客户端。
    """
    _state.store = store
    _state.autosave = not no_autosave
    if embedding_model:
        import os

        os.environ["MEMENTO_EMBEDDING_MODEL"] = embedding_model
    mcp.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="python -m memento.mcp_server",
        description="Memento MCP 服务",
    )
    p.add_argument(
        "--store",
        default=None,
        help="存储目录（默认 data/memento_store / $MEMENTO_STORE）",
    )
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--no-autosave", action="store_true", help="禁用写操作后自动落盘")
    args = p.parse_args()
    run_mcp(
        store=args.store,
        embedding_model=args.embedding_model,
        no_autosave=args.no_autosave,
    )
