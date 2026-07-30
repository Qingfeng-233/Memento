"""
对话 → 记忆切分辅助

把一轮（或多轮）对话切成可存入 Memento 的记忆节点。
当前用粗切策略：一整条 Q/A 存一个节点，和项目现有数据格式
（data/testtxt.txt 的【用户提问】...【AI 回答】... / memories.jsonl）一致。

未来可扩展为 LLM 细切（把长对话拆成多个语义片段），但先跑通闭环。
"""

from __future__ import annotations

import re
import time
from typing import Optional

from memento.api import Memento


def clean_reply_for_memory(reply: str, drop_mock_reply: bool = False) -> str:
    """清理不应进入长期记忆的 assistant 输出。"""
    if not reply:
        return ""
    if drop_mock_reply:
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\[Stream Error:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)

    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.lower().startswith("[mock"):
            continue
        if "Mock 回复" in stripped or "未接真实大模型" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def format_turn(user_text: str, reply: str) -> str:
    """把一轮对话格式化成记忆文本（和项目现有 Q/A 格式一致）。"""
    reply = reply.strip()
    if not reply:
        return f"用户问: {user_text}"
    return f"用户问: {user_text}\n回答: {reply}"


def ingest_turn(
    mem: Memento,
    user_text: str,
    reply: str,
    importance: float = 0.5,
    tags: Optional[list[str]] = None,
    source: str = "chat",
    drop_mock_reply: bool = False,
) -> str:
    """存一轮对话为记忆节点。

    粗切: 一整条 Q/A → 一个节点。返回 node_id。
    节点进入 pending 缓冲区，后续 build_index / 闲置巩固时才真正入索引。
    """
    text = format_turn(user_text, clean_reply_for_memory(reply, drop_mock_reply))
    node_id = mem.add_node(
        text=text,
        importance=importance,
        tags=tags or [],
        source=source,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return node_id


def pending_count(mem: Memento) -> int:
    """当前缓冲区里待 build 的节点数。"""
    return len(mem._pending_nodes)
