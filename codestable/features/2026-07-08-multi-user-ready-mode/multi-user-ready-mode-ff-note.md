---
doc_type: feature-ff-note
feature: multi-user-ready-mode
date: 2026-07-08
requirement:
tags: [chat-demo, multi-user, ownership, security]
---

## 做了什么

给聊天 demo 补齐最小 resource ownership：新写入 memory 带 owner 和 visibility 标签，检索、列表、观测和删除按当前 session owner 判断可见性。

## 改了哪些

- `apps/chat/app.py` — 新增 owner/visibility tag 和 `_node_visible_to_session()`，memory 默认 `visibility:private`，删除节点要求合法 `session_id` 且通过 ownership 校验。
- `apps/chat/static/index.html` — 删除记忆时携带当前 session。
- `scripts/test/test_chat_multi_user_ready_mode.py` — 验证 B 即使知道 A 的 node_id 也不能删除，A 可以删除自己的 node。
- `codestable/roadmap/production-chat-demo/` — 标记 `multi-user-ready-mode` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_multi_user_ready_mode.py` 验证 ownership 边界，并重跑 session 隔离、observability 与前端脚本语法检查。

## 顺手发现

- 当前仍是匿名 session 级 owner，不是账号系统；公开服务如果要长期运行，下一步应接入真实身份或部署层访问控制。
