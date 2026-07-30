---
doc_type: feature-ff-note
feature: session-isolation
date: 2026-07-08
requirement:
tags: [chat-demo, session, retrieval, memory]
---

## 做了什么

给聊天 demo 增加匿名 `session_id`，让不同浏览器会话的对话历史、长期记忆检索和侧栏记忆列表按 session 隔离。

## 改了哪些

- `apps/chat/app.py` — 新增 `/api/session`，持久化 `chat_sessions.json`，history 和 memory 写入携带 `session_id`，检索结果按 `session:<id>` tag 过滤。
- `apps/chat/static/index.html` — 前端用 localStorage 保存 session，聊天、history、reset-history、memories 请求携带当前 session。
- `scripts/test/test_chat_session_isolation.py` — 增加 A/B 两个 session 的 history 和 retrieval 隔离验证。
- `scripts/test/test_chat_turn_model.py` — 更新 turn 回归测试，使其走真实 session。
- `codestable/roadmap/production-chat-demo/` — 标记 `session-isolation` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_session_isolation.py` 验证 A 的 indexed memory 不会被 B 检索到，并重跑 `test_chat_turn_model.py` 与 `test_stable_memory_ids.py`。

## 顺手发现

- 现阶段隔离依赖 memory tags，而不是独立数据库查询条件；后续 `store-abstraction-v2` 应把 MemoryRecord 变成一等持久化记录。
