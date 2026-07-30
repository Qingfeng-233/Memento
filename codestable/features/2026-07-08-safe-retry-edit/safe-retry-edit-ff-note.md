---
doc_type: feature-ff-note
feature: safe-retry-edit
date: 2026-07-08
requirement:
tags: [chat-demo, retry, edit, turn-id, memory]
---

## 做了什么

把聊天 demo 的 retry/edit 从“删除最后一轮”改为按 `turn_id` 标记 superseded，旧 turn 和旧 memory 不物理删除，但默认不展示、不进 prompt、不参与检索。

## 改了哪些

- `apps/chat/app.py` — `replace_last` 优先使用 `retry_of_turn_id` / `edit_of_turn_id`，旧 history 和 memory 标记 `superseded`，新 turn 记录 supersede 来源。
- `memento/api.py` — pending node build 进图时保留 `status` 和 `superseded_by`，让未巩固旧记忆也能正确失效。
- `scripts/test/test_chat_safe_retry_edit.py` — 验证旧 turn 内部留存、外部 history 隐藏、旧 node 不再被 query 返回。
- `scripts/test/test_chat_turn_model.py` — 更新 retry 预期，旧 pending memory 标记 superseded 而不是删除。

## 怎么验证的

运行 `python scripts/test/test_chat_safe_retry_edit.py` 验证 supersede 状态和检索过滤；同时重跑 turn、session、cleaner 和 stable id 回归。

## 顺手发现

- 前端 retry/edit 仍会在 DOM 上移除旧 turn 来保持编辑体验；服务端已保留审计轨迹，后续 observability 面板可以展示 superseded 记录。
