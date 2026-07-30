---
doc_type: feature-ff-note
feature: turn-model
date: 2026-07-08
requirement:
tags: [chat-demo, conversation, turn-id, memory]
---

## 做了什么

给聊天 demo 的每轮对话引入后端生成的 `turn_id`，并把 SSE 响应、前端 turn、history 记录和写入的记忆 `node_id` 绑定起来。

## 改了哪些

- `apps/chat/app.py` — 扩展 `/api/chat` 协议，返回 `turn_id`，history 持久化 `turn_id`、`node_id`、`status` 和 supersede 信息。
- `apps/chat/static/index.html` — 前端保存服务端 `turn_id` / `node_id`，retry/edit 请求携带原始 turn。
- `scripts/test/test_chat_turn_model.py` — 增加 SSE turn 一致性和 retry 后 pending memory 绑定验证。
- `codestable/roadmap/production-chat-demo/` — 标记 `turn-model` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_turn_model.py` 验证普通聊天和 retry 的 `turn_id` / `node_id` 绑定；同时重跑 `python scripts/test/test_stable_memory_ids.py` 确认稳定 memory ID 未回退。

## 顺手发现

- `replace_last` 仍然是“删除最后一轮”的兼容路径，只是优先用 history 中的 `node_id` 删除记忆；真正按任意 `turn_id` 精确 supersede 要在 `safe-retry-edit` 完成。
