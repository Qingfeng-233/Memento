---
doc_type: audit-finding
audit: 2026-07-08-chat-demo-memory
finding_id: "bug-03"
nature: bug
severity: P1
confidence: high
suggested_action: cs-issue
status: open
---

# Finding 03：删除节点后按数量生成 ID 会撞已有节点，新记忆静默丢失

## 速答

`Memento.add_node()` 用 `len(graph.nodes) + len(_pending_nodes)` 生成新 ID。删除中间节点后，计数会回退到已有 ID，`add_node()` 发现 ID 已存在就直接返回，导致新对话没有进入 pending。

## 关键证据

- `memento/api.py:166` — 默认 `node_id = f"mem_{len(self.graph.nodes) + len(self._pending_nodes):06d}"`。
- `memento/api.py:169` — 如果 `node_id in self.graph.nodes`，直接 `return node_id`，没有生成下一个可用 ID，也没有报错。
- `apps/chat/app.py:451` — 前端可删除任意节点。
- 实测：创建 `mem_000000` 到 `mem_000003`，删除 `mem_000001` 后新增对话，返回 `new_id = mem_000003`；`pending` 仍为空，新记忆没有被写入。

## 影响

用户删除记忆后继续聊天，后续某些对话会静默丢失，不进入长期记忆。这个问题会直接破坏“记忆系统是否可靠”的基础判断。

## 修复方向

ID 生成应改成单调计数器、UUID/ULID，或基于持久化 `next_node_seq`；并且 `add_node()` 遇到自动 ID 冲突时应继续探测或抛出异常。

## 建议动作

`cs-issue`，因为这是底层写入路径的高置信 bug。
