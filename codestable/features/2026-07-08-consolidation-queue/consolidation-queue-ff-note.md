---
doc_type: feature-ff-note
feature: consolidation-queue
date: 2026-07-08
requirement:
tags: [chat-demo, consolidation, memory-lock, job]
---

## 做了什么

把聊天 demo 的手动/闲置巩固统一成串行 job，并用同一把 Memento 写锁协调聊天入库、删除、清空和巩固。

## 改了哪些

- `apps/chat/app.py` — 增加 `memory_lock`、`_run_consolidation_job()`，`/api/consolidate` 返回 job 状态并写入 consolidation log。
- `apps/chat/app.py` — 聊天入库、memory 删除、reset-memory 和 `_consolidate()` 使用同一写锁，降低并发修改 Memento 的风险。
- `scripts/test/test_chat_consolidation_queue.py` — 覆盖手动巩固 job、日志记录和 pending 入索引。
- `codestable/roadmap/production-chat-demo/` — 标记 `consolidation-queue` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_consolidation_queue.py` 验证 job 返回、日志和 build_index；同时重跑 safe retry 与 session 隔离回归确认锁调整未破坏对话入库。

## 顺手发现

- 当前实现是单进程串行 job，不是跨 worker 队列；如果后续部署多 worker，需要把 job 和锁迁移到外部存储或单独 worker。
