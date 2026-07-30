---
doc_type: feature-ff-note
feature: store-abstraction-v2
date: 2026-07-08
requirement:
tags: [chat-demo, store, persistence, json]
---

## 做了什么

新增 `JsonChatStore`，把聊天 demo 的 session、history 和 consolidation job JSON 持久化从 `app.py` 收口到独立 store 层。

## 改了哪些

- `apps/chat/store.py` — 新增 JSON ChatStore，实现 session/history/job 的 load/save/append。
- `apps/chat/app.py` — `ChatState` 使用 `JsonChatStore` 初始化和保存运行态，consolidation job 现在持久化到 `chat_consolidation_jobs.json`。
- `scripts/test/test_chat_store_abstraction.py` — 验证 app 重启后 session、history 和 job 能恢复。
- `codestable/roadmap/production-chat-demo/` — 标记 `store-abstraction-v2` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_store_abstraction.py` 验证持久化恢复，并重跑 auth、consolidation、safe retry、session 隔离回归。

## 顺手发现

- LLM settings 仍由 `apps/chat/llm.py` 的配置文件负责；后续若迁数据库，应再把模型设置纳入统一 ChatStore。
