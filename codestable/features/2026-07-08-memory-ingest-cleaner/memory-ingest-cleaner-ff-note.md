---
doc_type: feature-ff-note
feature: memory-ingest-cleaner
date: 2026-07-08
requirement:
tags: [chat-demo, memory, ingest, cleaner]
---

## 做了什么

给聊天入库链路增加 assistant 回复清洗，避免推理段、流式错误和 mock 调试文本进入长期记忆。

## 改了哪些

- `apps/chat/ingest.py` — 增加 `clean_reply_for_memory()`，`format_turn()` 支持无 assistant 回复时只保存用户问题。
- `apps/chat/app.py` — 调用 `ingest_turn()` 时按 LLM mock 状态决定是否丢弃 mock assistant 回复。
- `scripts/test/test_chat_memory_ingest_cleaner.py` — 覆盖 `<think>`、`[Stream Error]` 和 mock 回复清洗。
- `codestable/roadmap/production-chat-demo/` — 标记 `memory-ingest-cleaner` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_memory_ingest_cleaner.py` 验证 pending memory 只保留用户问题，不含 think、Mock 或未接模型提示；同时重跑 session、turn 和 stable id 回归。

## 顺手发现

- mock 模式现在只写用户问题，适合验证记忆闭环；真实回答质量评估必须切到真实 LLM。
