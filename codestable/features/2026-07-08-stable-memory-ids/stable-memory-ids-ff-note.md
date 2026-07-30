---
doc_type: feature-ff-note
feature: stable-memory-ids
date: 2026-07-08
requirement:
tags: [memory, node-id, chat-demo]
---

## 做了什么

修复默认 `node_id` 生成策略，避免删除中间节点后继续对话时复用已有 ID，导致新记忆静默丢失。

## 改了哪些

- `memento/api.py` — 增加 `_next_node_seq`、默认 ID 分配、显式 ID 序列推进和 meta 持久化恢复。
- `scripts/test/test_stable_memory_ids.py` — 增加删除后新增和 live add 的回归验证。
- `codestable/roadmap/production-chat-demo/` — 标记 roadmap 第一项完成。

## 怎么验证的

运行 `python scripts/test/test_stable_memory_ids.py` 验证删除 `mem_000001` 后新增得到 `mem_000004`，实时新增得到 `mem_000005`。

## 顺手发现

- `apps/chat/app.py` 的删除逻辑仍然直接操作 graph 和 `_id_map`，后续 `safe-retry-edit` / `store-abstraction-v2` 应统一收口删除语义。
