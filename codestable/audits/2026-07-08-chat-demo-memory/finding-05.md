---
doc_type: audit-finding
audit: 2026-07-08-chat-demo-memory
finding_id: "security-05"
nature: security
severity: P2
confidence: high
suggested_action: cs-refactor
status: open
---

# Finding 05：长期记忆存储了原始 assistant 回复，包含 `<think>` 思考段

## 速答

前端会把 `<think>...</think>` 折叠展示，但后端存长期记忆时使用完整 `full_reply`，导致思考段、调试文本或模型内部推理样式被写入长期记忆。

## 关键证据

- `apps/chat/static/index.html:773` — 前端只在展示层解析 `<think>` 和 `</think>`。
- `apps/chat/app.py:418` — 后端把流式输出拼成 `full_reply`。
- `apps/chat/app.py:421` — `ingest_turn(state.mem, user_text, full_reply, ...)` 原样存入长期记忆。
- 实测：mock 回复包含 `<think>`；发一轮消息后，`state.mem._pending_nodes[0]["text"]` 中包含 `<think>` 原文。

## 影响

检索质量会被重复的“思考链/Mock 思考链/分析步骤”污染；真实模型若输出 `<think>`，还会把用户不一定想长期保存的推理文本存进记忆库。

## 修复方向

在 ingest 前做 assistant 回复清洗：去掉 `<think>` 段、流式错误尾巴、UI 调试文本；必要时只存“用户问题 + 可见最终回答 + 结构化摘要”。

## 建议动作

`cs-refactor`，因为这是存储边界和数据清洗职责的重构问题。
