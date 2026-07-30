"""
Memento Chat — 后端

带记忆的对话 Web 应用后端。对话循环:
  用户输入 → 检索相关记忆 → 拼进 prompt → 调 LLM → 回复
  → 自动存这轮 Q/A → 重置闲置计时器

后台线程: 闲置 N 分钟无对话时自动巩固
  (build_index → build_concept_graph(可选) → trigger_sleep)

启动:
  python -m apps.chat.app [--host 127.0.0.1] [--port 8080]
  # 或
  python -m apps.chat.app --model deepseek-ai/DeepSeek-V3.2

默认复用 .env 里的 OPENCODE_API_* / SILICONFLOW_API_*，
只需指定 --model 即可接真 LLM；不指定则走 mock 模式先跑通闭环。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memento import store as store_mod
from memento.api import Memento
from apps.chat import llm as llm_mod
from apps.chat import ingest as ingest_mod
from apps.chat.store import JsonChatStore


# ─── 配置 ──────────────────────────────────────────────────


class ChatSettings:
    store: Optional[str]
    embedding_model: Optional[str] = None
    llm: llm_mod.LLMConfig = None
    auth_token: Optional[str] = None
    # 检索
    retrieve_k: int = 5  # 每轮注入多少条记忆
    retrieve_seed_k: int = 15
    min_index_size: int = 3  # 记忆少于此数时不注入（避免噪音）
    # 闲置巩固
    idle_minutes: float = 10.0  # 闲置多久触发巩固
    idle_check_interval: float = 30.0  # 后台检查间隔(秒)
    enable_concept_graph: bool = False  # 闲置时是否构建概念图(需 keyatten 模型)
    # 历史
    history_turns: int = 6  # 拼进 prompt 的历史对话轮数


# ─── 应用状态 ──────────────────────────────────────────────


class ChatState:
    mem: Memento
    settings: ChatSettings
    history: list[dict]  # [{"role","content"}, ...]
    last_activity: float
    last_consolidation: float
    consolidation_busy: bool
    consolidation_log: list[dict]
    retrieval_log: list[dict]
    error_log: list[dict]
    lock: threading.Lock
    memory_lock: threading.Lock
    chat_store: JsonChatStore
    sessions: dict[str, dict]

    def __init__(self, mem: Memento, settings: ChatSettings):
        self.mem = mem
        self.settings = settings
        self.last_activity = time.time()
        self.last_consolidation = time.time()
        self.consolidation_busy = False
        self.lock = threading.Lock()
        self.memory_lock = threading.Lock()
        store_path = Path(store_mod.resolve_store(settings.store))
        self.chat_store = JsonChatStore(store_path)
        self.sessions = self.chat_store.load_sessions()
        self.history = self.chat_store.load_history()
        self.consolidation_log = self.chat_store.load_consolidation_jobs()
        self.retrieval_log = []
        self.error_log = []

    def save_sessions(self):
        self.chat_store.save_sessions(self.sessions)

    def create_session(self, client_label: Optional[str] = None) -> dict:
        return self.chat_store.create_session(self.sessions, client_label)

    def save_history(self):
        self.chat_store.save_history(self.history)

    def append_consolidation_job(self, job: dict):
        self.consolidation_log = self.chat_store.append_consolidation_job(job)

    def append_retrieval_event(self, event: dict):
        self.retrieval_log.append(event)
        if len(self.retrieval_log) > 50:
            self.retrieval_log = self.retrieval_log[-50:]

    def append_error_event(self, event: dict):
        self.error_log.append(event)
        if len(self.error_log) > 50:
            self.error_log = self.error_log[-50:]


# ─── 闲置巩固后台线程 ─────────────────────────────────────


def _consolidate(state: ChatState) -> dict:
    """执行一次巩固。返回日志字典。"""
    with state.memory_lock:
        mem = state.mem
        s = state.settings
        log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}

        # 1. build pending
        pending = ingest_mod.pending_count(mem)
        if pending > 0:
            added = mem.build_index()
            log["steps"].append(f"build_index: {added} 个节点入索引")
        else:
            log["steps"].append("build_index: 无 pending")

        # 2. 概念图（可选，需 keyatten 模型，较慢）
        if s.enable_concept_graph and mem._index_built and mem.graph.node_count > 0:
            try:
                info = mem.build_concept_graph(
                    max_concepts=300,
                    min_concept_energy=0.5,
                )
                log["steps"].append(
                    f"concept_graph: {info['concepts']} 概念, "
                    f"{info['event_concept_edges']} 事件-概念边"
                )
            except Exception as e:
                log["steps"].append(f"concept_graph 失败: {e}")

        # 3. 睡眠巩固
        report = mem.trigger_sleep()
        log["steps"].append(
            f"sleep: 边强化 {report.edges_strengthened}, "
            f"修剪 {report.edges_pruned}, 休眠 {report.nodes_dormant}"
        )

        store_mod.save(mem, s.store)
        state.last_consolidation = time.time()
        return log


def _run_consolidation_job(state: ChatState, trigger: str) -> dict:
    job = {
        "job_id": f"job_{uuid.uuid4().hex}",
        "trigger": trigger,
        "status": "running",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": _now_text(),
        "finished_at": None,
        "steps": [],
        "error": None,
    }
    try:
        log = _consolidate(state)
        job["steps"] = log.get("steps", [])
        job["status"] = "done"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        job["finished_at"] = _now_text()
        with state.lock:
            state.append_consolidation_job(job)
    return job


def _idle_watcher(state: ChatState):
    """后台线程: 闲置超时则巩固，循环检查。"""
    s = state.settings
    while True:
        time.sleep(s.idle_check_interval)
        with state.lock:
            idle = time.time() - state.last_activity
            pending = ingest_mod.pending_count(state.mem)
            # 闲置超时 且 有待巩固内容(新 pending) 且 不在忙
            if (
                idle >= s.idle_minutes * 60
                and pending > 0
                and not state.consolidation_busy
            ):
                state.consolidation_busy = True
            else:
                continue
        try:
            _run_consolidation_job(state, "idle")
        finally:
            with state.lock:
                state.consolidation_busy = False


# ─── FastAPI 应用 ─────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    text: str
    replace_last: bool = False  # 编辑/重试最后一条用户消息时为 True
    retry_of_turn_id: Optional[str] = None
    edit_of_turn_id: Optional[str] = None


class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    client_label: Optional[str] = None


class SettingsRequest(BaseModel):
    api_base: Optional[str] = None
    api_key: Optional[str] = None  # 为空/****表示不修改
    model: Optional[str] = None


class ModelsRequest(BaseModel):
    api_base: Optional[str] = None  # 不传则用当前配置
    api_key: Optional[str] = None


def _new_turn_id() -> str:
    return f"turn_{uuid.uuid4().hex}"


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _turn_history_entries(
    session_id: str,
    turn_id: str,
    user_text: str,
    assistant_text: str,
    node_id: str,
    supersedes_turn_id: Optional[str] = None,
) -> list[dict]:
    created_at = _now_text()
    base = {
        "turn_id": turn_id,
        "session_id": session_id,
        "status": "done",
        "created_at": created_at,
        "updated_at": created_at,
    }
    if supersedes_turn_id:
        base["supersedes_turn_id"] = supersedes_turn_id

    return [
        {
            **base,
            "role": "user",
            "content": user_text,
        },
        {
            **base,
            "role": "assistant",
            "content": assistant_text,
            "node_id": node_id,
        },
    ]


def _prompt_history(history: list[dict]) -> list[dict]:
    messages = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def _session_tag(session_id: str) -> str:
    return f"session:{session_id}"


def _owner_tag(owner_id: Optional[str]) -> str:
    return f"owner:{owner_id or ''}"


def _history_for_session(
    history: list[dict],
    session_id: str,
    include_superseded: bool = False,
) -> list[dict]:
    items = [item for item in history if item.get("session_id") == session_id]
    if include_superseded:
        return items
    return [item for item in items if item.get("status") != "superseded"]


def _mark_memory_superseded(
    mem: Memento,
    node_id: str,
    superseded_by: Optional[str] = None,
) -> None:
    node = mem.graph.get_node(node_id)
    if node:
        node.status = "superseded"
        node.superseded_by = superseded_by
    for pending in mem._pending_nodes:
        if pending.get("id") == node_id:
            pending["status"] = "superseded"
            pending["superseded_by"] = superseded_by


def _node_has_session(node, session_id: str) -> bool:
    return bool(node and _session_tag(session_id) in (node.tags or []))


def _node_visible_to_session(node, session: Optional[dict]) -> bool:
    if not node or not session:
        return False
    tags = node.tags or []
    visibility = "private"
    for tag in tags:
        if tag.startswith("visibility:"):
            visibility = tag.split(":", 1)[1]
            break
    if visibility == "public":
        return True
    if visibility == "session":
        return _session_tag(session["session_id"]) in tags
    owner_id = session.get("owner_id")
    if _owner_tag(owner_id) in tags:
        return True
    # 兼容旧数据：没有 owner/visibility 时按 session tag 判断。
    return _session_tag(session["session_id"]) in tags


def _filter_memories_for_session(
    mem: Memento,
    memories: list[dict],
    session: dict,
    limit: int,
) -> list[dict]:
    visible = []
    for item in memories:
        node = mem.graph.get_node(item["id"])
        if _node_visible_to_session(node, session):
            visible.append(item)
        if len(visible) >= limit:
            break
    return visible


def _turns_for_observability(history: list[dict], session_id: Optional[str]) -> list[dict]:
    turns: dict[str, dict] = {}
    for item in history:
        if session_id and item.get("session_id") != session_id:
            continue
        turn_id = item.get("turn_id")
        if not turn_id:
            continue
        turn = turns.setdefault(
            turn_id,
            {
                "turn_id": turn_id,
                "session_id": item.get("session_id"),
                "status": item.get("status", "done"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "user_preview": "",
                "assistant_preview": "",
                "node_id": None,
                "supersedes_turn_id": item.get("supersedes_turn_id"),
                "superseded_by_turn_id": item.get("superseded_by_turn_id"),
            },
        )
        if item.get("role") == "user":
            turn["user_preview"] = (item.get("content") or "")[:120]
        elif item.get("role") == "assistant":
            turn["assistant_preview"] = (item.get("content") or "")[:120]
            turn["node_id"] = item.get("node_id")
        if item.get("status") == "superseded":
            turn["status"] = "superseded"
            turn["superseded_by_turn_id"] = item.get("superseded_by_turn_id")
    return sorted(turns.values(), key=lambda t: t.get("created_at") or "", reverse=True)


def _memory_events_for_observability(
    mem: Memento,
    session: Optional[dict],
) -> list[dict]:
    events = []
    for pending in mem._pending_nodes:
        tags = pending.get("tags", [])
        if session and not _node_visible_to_session(
            type("PendingNode", (), {"tags": tags})(),
            session,
        ):
            continue
        events.append(
            {
                "node_id": pending.get("id"),
                "status": pending.get("status", "active"),
                "stage": "pending",
                "source": pending.get("source"),
                "tags": tags,
                "text_preview": (pending.get("text") or "")[:120],
                "superseded_by": pending.get("superseded_by"),
            }
        )
    for node in mem.graph.nodes.values():
        if session and not _node_visible_to_session(node, session):
            continue
        events.append(
            {
                "node_id": node.id,
                "status": node.status,
                "stage": "indexed",
                "source": node.source,
                "tags": node.tags,
                "text_preview": (node.text or "")[:120],
                "superseded_by": node.superseded_by,
            }
        )
    return sorted(events, key=lambda e: e.get("node_id") or "", reverse=True)


def create_app(settings: ChatSettings) -> FastAPI:
    mem = store_mod.load_or_create(settings.store, settings.embedding_model)
    state = ChatState(mem, settings)

    app = FastAPI(title="Memento Chat", version="0.1.0")
    app.state.chat = state

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def require_token(x_memento_token: Optional[str] = None) -> None:
        token = settings.auth_token or ""
        if token and x_memento_token != token:
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/")
    def index():
        idx = static_dir / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return JSONResponse({"error": "index.html not found"}, 404)

    @app.get("/api/config")
    def get_config():
        s = settings
        return {
            "llm_mock": s.llm.mock,
            "llm_model": s.llm.model or "(mock)",
            "store": store_mod.resolve_store(s.store),
            "retrieve_k": s.retrieve_k,
            "idle_minutes": s.idle_minutes,
            "enable_concept_graph": s.enable_concept_graph,
            "auth_required": bool(s.auth_token),
        }

    @app.post("/api/session")
    def create_session(req: SessionRequest):
        with state.lock:
            if req.session_id and req.session_id in state.sessions:
                return state.sessions[req.session_id]
            return state.create_session(req.client_label)

    # ─── LLM 设置（网页可编辑 + 持久化）──────────────────
    @app.get("/api/settings")
    def get_settings(x_memento_token: Optional[str] = Header(default=None)):
        require_token(x_memento_token)
        s = settings.llm
        # api_key 脱敏回传（只露末4位），避免明文暴露到前端
        key = s.api_key or ""
        masked = key[:-4] + "****" if len(key) > 4 else ("****" if key else "")
        return {
            "api_base": s.api_base,
            "api_key_masked": masked,
            "api_key_set": bool(key),
            "model": s.model,
            "mock": s.mock,
        }

    @app.post("/api/settings")
    def save_settings(
        req: SettingsRequest,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        require_token(x_memento_token)
        s = settings
        cur = {
            "api_base": s.llm.api_base,
            "api_key": s.llm.api_key,
            "model": s.llm.model,
        }
        # 只更新非空字段；api_key 为空或全 **** 表示保持原值
        if req.api_base is not None and req.api_base.strip():
            cur["api_base"] = req.api_base.strip()
        if req.api_key and not set(req.api_key) <= {"*"}:
            cur["api_key"] = req.api_key.strip()
        if req.model is not None and req.model.strip():
            cur["model"] = req.model.strip()

        # 写文件 + 热重建 LLMConfig
        llm_mod.save_config_to_file(cur)
        s.llm = llm_mod.load_llm_config()
        return {
            "ok": True,
            "mock": s.llm.mock,
            "model": s.llm.model or "(mock)",
            "api_base": s.llm.api_base,
        }

    @app.post("/api/models")
    def list_models(
        req: ModelsRequest,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        require_token(x_memento_token)
        base = (req.api_base or settings.llm.api_base or "").strip()
        # api_key: 请求传入优先，否则用当前配置；全 **** 视为未改用当前
        key = req.api_key
        if not key or set(key) <= {"*"}:
            key = settings.llm.api_key or ""
        try:
            models = llm_mod.fetch_models(base, key)
            return {"ok": True, "models": models}
        except Exception as e:
            return {"ok": False, "error": str(e), "models": []}

    @app.get("/api/stats")
    def get_stats():
        return {
            "memento": state.mem.stats,
            "pending": ingest_mod.pending_count(state.mem),
            "idle_seconds": round(time.time() - state.last_activity, 0),
            "consolidation_busy": state.consolidation_busy,
            "last_consolidation": state.last_consolidation,
            "history_len": len(state.history),
        }

    @app.get("/api/memories")
    def get_memories(
        limit: int = 20,
        session_id: Optional[str] = None,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        require_token(x_memento_token)
        """最近 N 条记忆（按 id 倒序近似最近）。"""
        nodes = sorted(
            state.mem.graph.nodes.values(),
            key=lambda n: n.id,
            reverse=True,
        )
        session = state.sessions.get(session_id or "")
        if session_id:
            nodes = [n for n in nodes if _node_visible_to_session(n, session)]
        nodes = nodes[:limit]
        return [store_mod.node_to_dict(n) for n in nodes]

    @app.get("/api/consolidation-log")
    def get_consolidation_log(
        x_memento_token: Optional[str] = Header(default=None),
    ):
        require_token(x_memento_token)
        return state.consolidation_log[-10:]

    @app.get("/api/observability")
    def get_observability(
        session_id: Optional[str] = None,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        require_token(x_memento_token)
        session_filter = session_id if session_id in state.sessions else None
        return {
            "session_id": session_filter,
            "turns": _turns_for_observability(state.history, session_filter)[:20],
            "memory_events": _memory_events_for_observability(
                state.mem,
                state.sessions.get(session_filter or ""),
            )[:20],
            "jobs": state.consolidation_log[-10:],
            "retrievals": [
                event
                for event in state.retrieval_log[-20:]
                if not session_filter or event.get("session_id") == session_filter
            ][-10:],
            "errors": [
                event
                for event in state.error_log[-20:]
                if not session_filter or event.get("session_id") == session_filter
            ][-10:],
        }

    @app.post("/api/consolidate")
    def manual_consolidate(
        x_memento_token: Optional[str] = Header(default=None),
    ):
        """手动触发巩固。"""
        require_token(x_memento_token)
        with state.lock:
            if state.consolidation_busy:
                return {"ok": False, "busy": True}
            state.consolidation_busy = True
        try:
            job = _run_consolidation_job(state, "manual")
            return {
                "ok": job["status"] == "done",
                "job": job,
                "log": job,
                "error": job.get("error"),
            }
        finally:
            with state.lock:
                state.consolidation_busy = False

    @app.post("/api/chat")
    def chat(
        req: ChatRequest,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        """一轮对话: 检索 → 注入 → 调 LLM → 回复 → 存这轮 (流式 SSE 版)。

        replace_last=True: 编辑/重试时按 turn_id 标记旧 turn 和旧 memory 为
        superseded，再用 req.text 重新走流程。
        """
        require_token(x_memento_token)
        s = settings
        with state.lock:
            state.last_activity = time.time()

        user_text = req.text.strip()
        if not user_text:
            def empty_gen():
                yield f"data: {json.dumps({'type': 'error', 'turn_id': None, 'error': 'empty input'}, ensure_ascii=False)}\n\n"
            return StreamingResponse(empty_gen(), media_type="text/event-stream")

        session = state.sessions.get(req.session_id or "")
        if not session:
            def invalid_session_gen():
                state.append_error_event(
                    {
                        "time": _now_text(),
                        "session_id": req.session_id,
                        "turn_id": None,
                        "error": "invalid session",
                    }
                )
                yield f"data: {json.dumps({'type': 'error', 'turn_id': None, 'error': 'invalid session'}, ensure_ascii=False)}\n\n"
            return StreamingResponse(invalid_session_gen(), media_type="text/event-stream")

        session_id = session["session_id"]
        turn_id = _new_turn_id()
        supersedes_turn_id = req.edit_of_turn_id or req.retry_of_turn_id

        # 编辑/重试：按 turn 标记旧历史和旧记忆为 superseded。
        superseded_turn_ids = set()
        superseded_node_ids = []
        if req.replace_last:
            with state.lock:
                session_history = _history_for_session(
                    state.history,
                    session_id,
                    include_superseded=True,
                )
                active_history = [
                    item
                    for item in session_history
                    if item.get("status") != "superseded"
                ]
                target_turn_id = supersedes_turn_id
                if not target_turn_id and active_history:
                    target_turn_id = active_history[-1].get("turn_id")
                removed_entries = [
                    item
                    for item in session_history
                    if target_turn_id and item.get("turn_id") == target_turn_id
                ]
                for entry in reversed(removed_entries):
                    if entry.get("role") == "assistant" and entry.get("node_id"):
                        superseded_node_ids.append(entry["node_id"])
                for entry in removed_entries:
                    entry["status"] = "superseded"
                    entry["updated_at"] = _now_text()
                    if entry.get("turn_id"):
                        superseded_turn_ids.add(entry["turn_id"])
                state.save_history()
            with state.memory_lock:
                for old_node_id in superseded_node_ids:
                    _mark_memory_superseded(state.mem, old_node_id)

        # 1. 检索相关记忆（索引建好且记忆足够多才注入，否则纯对话）
        memories = []
        if state.mem._index_built and state.mem.graph.node_count >= s.min_index_size:
            try:
                memories = state.mem.query(
                    user_text,
                    k=max(s.retrieve_k * 5, s.retrieve_k + 20),
                    seed_k=s.retrieve_seed_k,
                )
                memories = _filter_memories_for_session(
                    state.mem,
                    memories,
                    session,
                    s.retrieve_k,
                )
            except Exception:
                memories = []

        # 2. 拼 prompt（system + 记忆 + 历史 + 本轮）
        session_history = _history_for_session(state.history, session_id)
        history = _prompt_history(session_history[-(s.history_turns * 2) :])
        messages = llm_mod.build_chat_prompt(
            user_text,
            memories,
            s.llm,
            history=history,
        )

        # 3. 流式生成器包装
        def sse_generator():
            # a. 发送 info 事件，回传所引用的记忆和 mock 状态
            import memento.store as st
            mem_dicts = [st.node_to_dict(state.mem.graph.get_node(m["id"])) if state.mem.graph.get_node(m["id"]) else m for m in memories]
            # 保留原 memories 的 score
            for md, orig in zip(mem_dicts, memories):
                if isinstance(md, dict) and "score" not in md:
                    md["score"] = orig.get("score", 0.0)
            
            info_data = {
                "type": "info",
                "turn_id": turn_id,
                "memories_used": mem_dicts,
                "mock": s.llm.mock,
            }
            state.append_retrieval_event(
                {
                    "time": _now_text(),
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "query_preview": user_text[:120],
                    "memory_ids": [m.get("id") for m in memories],
                    "memory_count": len(memories),
                }
            )
            yield f"data: {json.dumps(info_data, ensure_ascii=False)}\n\n"

            # b. 开始流式调用大模型，实时 yield 文本
            full_reply_parts = []
            try:
                for chunk in llm_mod.chat_stream(messages, s.llm):
                    full_reply_parts.append(chunk)
                    chunk_data = {
                        "type": "content",
                        "turn_id": turn_id,
                        "content": chunk,
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            except Exception as e:
                state.append_error_event(
                    {
                        "time": _now_text(),
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "error": f"LLM 调用失败: {e}",
                    }
                )
                err_data = {
                    "type": "error",
                    "turn_id": turn_id,
                    "error": f"LLM 调用失败: {e}",
                }
                yield f"data: {json.dumps(err_data, ensure_ascii=False)}\n\n"
                return

            full_reply = "".join(full_reply_parts)

            # c. 将这轮完整对话存为长期记忆
            with state.memory_lock:
                node_id = ingest_mod.ingest_turn(
                    state.mem,
                    user_text,
                    full_reply,
                    tags=[
                        f"turn:{turn_id}",
                        _session_tag(session_id),
                        _owner_tag(session.get("owner_id")),
                        "visibility:private",
                    ],
                    source="chat",
                    drop_mock_reply=s.llm.mock,
                )
                for old_node_id in superseded_node_ids:
                    _mark_memory_superseded(state.mem, old_node_id, node_id)

            # d. 永久追加历史到 state.history
            with state.lock:
                for entry in state.history:
                    if entry.get("turn_id") in superseded_turn_ids:
                        entry["superseded_by_turn_id"] = turn_id
                        entry["superseded_by_node_id"] = node_id
                        entry["updated_at"] = _now_text()
                state.history.extend(
                    _turn_history_entries(
                        session_id,
                        turn_id,
                        user_text,
                        full_reply,
                        node_id,
                        supersedes_turn_id=supersedes_turn_id,
                    )
                )
                state.save_history()

            # e. 发送 done 结束事件，回传 node_id
            done_data = {
                "type": "done",
                "turn_id": turn_id,
                "node_id": node_id,
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    @app.post("/api/reset-history")
    def reset_history(
        session_id: Optional[str] = None,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        """清空对话历史（不影响记忆库）。"""
        require_token(x_memento_token)
        with state.lock:
            if session_id:
                state.history = [
                    item
                    for item in state.history
                    if item.get("session_id") != session_id
                ]
            else:
                state.history = []
            state.save_history()
        return {"ok": True}

    @app.delete("/api/node/{node_id}")
    def delete_node(
        node_id: str,
        session_id: Optional[str] = None,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        """删除单个记忆节点。"""
        require_token(x_memento_token)
        session = state.sessions.get(session_id or "")
        if not session:
            return JSONResponse({"error": "invalid session"}, status_code=400)
        with state.memory_lock:
            node = state.mem.graph.get_node(node_id)
            pending = next(
                (p for p in state.mem._pending_nodes if p.get("id") == node_id),
                None,
            )
            if node and not _node_visible_to_session(node, session):
                return JSONResponse({"error": "forbidden"}, status_code=403)
            if pending:
                pending_node = type("PendingNode", (), {"tags": pending.get("tags", [])})()
                if not _node_visible_to_session(pending_node, session):
                    return JSONResponse({"error": "forbidden"}, status_code=403)
            # 1. 从图中弹出节点
            state.mem.graph.nodes.pop(node_id, None)
            
            # 2. 清理邻接表和相关的边，更新邻居的 edge_count
            neighbors = state.mem.graph._adjacency.pop(node_id, {})
            for neighbor_id in neighbors:
                if neighbor_id in state.mem.graph._adjacency:
                    state.mem.graph._adjacency[neighbor_id].pop(node_id, None)
                    if neighbor_id in state.mem.graph.nodes:
                        state.mem.graph.nodes[neighbor_id].edge_count = len(state.mem.graph._adjacency[neighbor_id])
            
            # 3. 从向量索引映射中设为空字符串，防止索引移位错乱
            if node_id in state.mem.vector_index._id_map:
                idx = state.mem.vector_index._id_map.index(node_id)
                state.mem.vector_index._id_map[idx] = ""
            
            # 4. 从 pending 缓冲队列中清除
            state.mem._pending_nodes = [
                p for p in state.mem._pending_nodes if p["id"] != node_id
            ]
            
            # 5. 立刻落盘保存以确保持久化一致
            store_mod.save(state.mem, settings.store)
            
        return {"ok": True}

    @app.post("/api/reset-memory")
    def reset_memory(x_memento_token: Optional[str] = Header(default=None)):
        """彻底重置/清空记忆库（重新来过）。"""
        require_token(x_memento_token)
        with state.memory_lock:
            # 1. 创建全新空的 Memento 实例
            model = store_mod.resolve_embedding_model(settings.embedding_model, settings.store)
            state.mem = Memento(embedding_model=model)
            
            # 2. 物理删除磁盘上的存储文件（保留 store_meta.json）
            store_path = Path(store_mod.resolve_store(settings.store))
            keep_files = {
                "store_meta.json",
                state.chat_store.sessions_path.name,
                state.chat_store.jobs_path.name,
            }
            if store_path.exists():
                for f in store_path.glob("*"):
                    if f.name not in keep_files:
                        try:
                            f.unlink()
                        except Exception:
                            pass
            
            # 3. 空实例立刻保存落盘
            store_mod.save(state.mem, settings.store)
            
        with state.lock:
            # 4. 重置对话历史
            state.history = []
            state.save_history()
            
        return {"ok": True}

    @app.get("/api/history")
    def get_history(
        session_id: Optional[str] = None,
        x_memento_token: Optional[str] = Header(default=None),
    ):
        """获取当前会话对话历史。"""
        require_token(x_memento_token)
        if not session_id:
            return []
        if session_id not in state.sessions:
            return JSONResponse({"error": "invalid session"}, status_code=404)
        return _history_for_session(state.history, session_id)

    # 启动闲置巩固后台线程
    t = threading.Thread(target=_idle_watcher, args=(state,), daemon=True)
    t.start()

    return app


# ─── CLI 入口 ─────────────────────────────────────────────


def make_settings(args) -> ChatSettings:
    s = ChatSettings()
    s.store = args.store
    s.embedding_model = args.embedding_model
    s.auth_token = args.token or os.environ.get("MEMENTO_CHAT_TOKEN", "")
    s.llm = llm_mod.load_llm_config(
        model=args.model,
        mock=True if args.mock else None,
    )
    s.retrieve_k = args.retrieve_k
    s.idle_minutes = args.idle_minutes
    s.enable_concept_graph = args.concepts
    s.history_turns = args.history_turns
    return s


def run(args=None):
    p = argparse.ArgumentParser(
        prog="python -m apps.chat.app",
        description="Memento Chat — 带记忆的对话 Web 应用",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--store", default=None, help="Memento 存储目录（默认 data/memento_store）"
    )
    p.add_argument(
        "--embedding-model", default=None, help="embedding 后端（默认 tfidf-svd）"
    )
    p.add_argument(
        "--model",
        default=None,
        help="LLM 模型名（如 deepseek-ai/DeepSeek-V3.2）。不指定则走 mock 模式",
    )
    p.add_argument("--mock", action="store_true", help="强制 mock 模式（不调 LLM）")
    p.add_argument("--token", default=None, help="启用 HTTP API token 保护")
    p.add_argument("--retrieve-k", type=int, default=5, help="每轮注入的记忆条数")
    p.add_argument(
        "--idle-minutes", type=float, default=10.0, help="闲置多少分钟触发自动巩固"
    )
    p.add_argument(
        "--concepts",
        action="store_true",
        help="闲置巩固时构建概念图（需 keyatten 模型，较慢）",
    )
    p.add_argument(
        "--history-turns", type=int, default=6, help="拼进 prompt 的历史对话轮数"
    )
    args = p.parse_args(args)

    import uvicorn

    settings = make_settings(args)
    app = create_app(settings)
    print(f"Memento Chat 启动: http://{args.host}:{args.port}")
    print(f"  LLM: {'mock' if settings.llm.mock else settings.llm.model}")
    print(f"  store: {store_mod.resolve_store(settings.store)}")
    print(f"  闲置巩固: {settings.idle_minutes} 分钟")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
