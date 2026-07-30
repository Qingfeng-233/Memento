---
doc_type: audit-finding
audit: 2026-07-08-chat-demo-memory
finding_id: "security-01"
nature: security
severity: P1
confidence: high
suggested_action: cs-issue
status: open
---

# Finding 01：全局 history / memory 导致多人对话互相泄露

## 速答

`apps/chat` 只有一个全局 `ChatState`，所有浏览器/用户共用同一份 `history` 和 `mem`，多人测试时 Bob 的 prompt 会包含 Alice 的历史，Bob 的检索也能召回 Alice 的私密记忆。

## 关键证据

- `apps/chat/app.py:200` — `create_app()` 只创建一个 `ChatState(mem, settings)`，挂到 `app.state.chat`。
- `apps/chat/app.py:379` — 每次聊天直接取 `state.history[-(s.history_turns * 2):]` 拼 prompt，没有 user/session 过滤。
- `apps/chat/app.py:367` — 检索直接对 `state.mem.query(...)` 全局记忆库执行，没有 namespace 过滤。
- 实测：Alice 先说 `blue-raven`，Bob 第二轮 prompt 中包含 Alice 的 `blue-raven`；巩固 4 条后，Bob 查询 `blue-raven` 返回了 Alice 的记忆。

## 影响

开源 demo 如果让多个人共用一个服务地址测试，会出现跨用户 prompt 泄露和长期记忆泄露。即便默认监听 `127.0.0.1`，只要用户用 `--host 0.0.0.0` 或反代给别人测，就会变成真实隐私问题。

## 修复方向

先明确产品模型：单用户本地工具就显式限制和提示；多人测试服务则引入 `session_id/user_id`，并让 history、memory、settings 至少按 session/user 隔离。

## 建议动作

`cs-issue`，因为这是明确可复现的安全边界缺失。
