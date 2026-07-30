---
doc_type: audit-finding
audit: 2026-07-08-chat-demo-memory
finding_id: "bug-02"
nature: bug
severity: P1
confidence: high
suggested_action: cs-issue
status: open
---

# Finding 02：`replace_last` 按全局最后一轮回滚，会删错用户和错节点

## 速答

编辑/重试没有携带 turn_id 或 node_id，后端只按全局最后一轮和最新 chat 节点回滚。多人交替对话时，Alice 重试自己的消息会回滚 Bob 的最近对话，并可能删除另一个已巩固节点。

## 关键证据

- `apps/chat/app.py:329` — 注释说明 `replace_last=True` 会移除“上一轮 user+assistant”。
- `apps/chat/app.py:344` — 触发条件仅为 `req.replace_last and len(state.history) >= 2`，没有校验这次请求对应哪一轮。
- `apps/chat/app.py:347` — 直接 `state.history = state.history[:-2]`，删除全局最后两条历史。
- `apps/chat/app.py:351` — 通过“最近一个 `source=chat` 的节点”猜测要删的节点，和被编辑的前端 turn 没有稳定关系。
- 实测：Alice/Alice/Bob/Bob/Bob-query 后，Alice 用 `replace_last=True`，历史中 Bob-query 被移除，graph 中最新已巩固节点也被删。

## 影响

多人测试下会删错人；单人快速编辑旧消息时也会删错轮次。由于向量索引和图节点不是事务更新，后续检索还可能出现索引残留或结果丢失。

## 修复方向

把 turn_id/node_id 明确返回给前端并随 retry/edit 提交；后端只允许回滚匹配的 turn，并把历史、pending、graph、vector id_map 作为一个一致性单元更新。

## 建议动作

`cs-issue`，因为这是已复现的数据一致性 bug。
