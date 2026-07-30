---
doc_type: feature-ff-note
feature: demo-auth-token
date: 2026-07-08
requirement:
tags: [chat-demo, auth, token, security]
---

## 做了什么

给聊天 demo 增加可选 token 保护；默认不配置时保持本地开发体验，配置后核心读写和危险接口必须携带 `X-Memento-Token`。

## 改了哪些

- `apps/chat/app.py` — 增加 `ChatSettings.auth_token`、`--token` / `MEMENTO_CHAT_TOKEN`，并保护 chat、history、memories、settings、models、consolidate、delete、reset 等接口。
- `apps/chat/static/index.html` — 前端从 localStorage 读取 token，请求自动带 header，401 时提示输入 token。
- `scripts/test/test_chat_demo_auth_token.py` — 覆盖无 token/错误 token 拒绝、正确 token 可聊天和读 history。
- `codestable/roadmap/production-chat-demo/` — 标记 `demo-auth-token` 完成。

## 怎么验证的

运行 `python scripts/test/test_chat_demo_auth_token.py` 验证 token 开启后的保护行为，并重跑 consolidation 与 cleaner 回归确认默认无 token 路径不受影响。

## 顺手发现

- 当前 token 是 demo 级共享密钥，不是账号系统；多人服务还需要 ownership 检查和更完整的部署约束。
