---
doc_type: audit-finding
audit: 2026-07-08-chat-demo-memory
finding_id: "security-04"
nature: security
severity: P1
confidence: medium
suggested_action: cs-issue
status: open
---

# Finding 04：缺少鉴权/访问边界，读写和清空接口可被任意本地 HTTP 调用

## 速答

聊天 demo 暴露了历史、记忆、设置、删除、清空等接口，但没有任何鉴权、CSRF 防护或本地访问限制检查。默认本地跑风险较低，一旦开放 host 给别人测试，风险立刻升高。

## 关键证据

- `apps/chat/app.py:231` — `/api/settings` 返回当前 LLM 配置状态和模型信息。
- `apps/chat/app.py:295` — `/api/memories` 返回最近记忆文本。
- `apps/chat/app.py:443` — `/api/reset-history` 清空全局对话历史。
- `apps/chat/app.py:451` — `/api/node/{node_id}` 删除任意记忆节点。
- `apps/chat/app.py:481` — `/api/reset-memory` 物理清空记忆库文件。
- `apps/chat/app.py:508` — `/api/history` 返回完整全局历史。

## 影响

如果测试者把服务绑定到 `0.0.0.0`、局域网或隧道，任何能访问 URL 的人都能读取历史/记忆、枚举本地配置状态、删除节点或清空记忆库。

## 修复方向

至少增加一个本地 demo token；危险写操作要求 token；文档明确默认只建议 `127.0.0.1`。如果目标是多人测试服务，需要完整的用户身份和资源归属校验。

## 建议动作

`cs-issue`，因为这是开源测试时容易被误用的安全边界问题。
