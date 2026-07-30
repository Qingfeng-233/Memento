---
doc_type: feature-ff-note
feature: observability-panel
date: 2026-07-08
requirement:
tags: [chat-demo, observability, debug, memory]
---

## 做了什么

给聊天 demo 增加观测 API 和侧栏调试面板，用于查看 turn、memory 写入、巩固 job、retrieval 和错误状态。

## 改了哪些

- `apps/chat/app.py` — 新增 `/api/observability`，记录最近 retrieval/error，聚合 turn、memory_events、jobs。
- `apps/chat/static/index.html` — 侧栏新增“调试观测”面板，展示当前 session 的 turn/memory/job/retrieval/error 摘要。
- `scripts/test/test_chat_observability_panel.py` — 覆盖 observability API 的 turn、memory、job、retrieval 输出。
- `codestable/roadmap/production-chat-demo/` — 标记 `observability-panel` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_observability_panel.py` 验证观测数据；用 `node --check` 检查前端脚本语法。

## 顺手发现

- retrieval 目前只记录最近进程内事件，重启后不恢复；如果要长期排查检索行为，应后续纳入 ChatStore。
