---
doc_type: audit-finding
audit: 2026-07-08-chat-demo-memory
finding_id: "bug-06"
nature: bug
severity: P2
confidence: medium
suggested_action: cs-issue
status: open
---

# Finding 06：巩固过程没有持有状态锁，可能与聊天写入并发修改同一 Memento

## 速答

`manual_consolidate()` 只在设置 busy 标记时持锁，真正的 `_consolidate()` 在锁外执行；聊天流完成后也会写 `state.mem`。两条路径可能同时改 `_pending_nodes`、vector_index、graph 和磁盘文件。

## 关键证据

- `apps/chat/app.py:312` — `manual_consolidate()` 进入锁只检查并设置 `consolidation_busy`。
- `apps/chat/app.py:317` — `_consolidate(state)` 在锁外执行。
- `apps/chat/app.py:421` — 聊天 SSE 结束后向同一个 `state.mem` 写入新节点。
- `apps/chat/app.py:429` — 聊天只在追加 history 时持锁，不保护 `ingest_turn()`。
- `apps/chat/app.py:110` — `_consolidate()` 内会调用 `build_index()`、`trigger_sleep()`、`store_mod.save()`，都是共享状态变更。

## 影响

单人低频 demo 不容易撞上；多人测试或真实 LLM 慢流式时，用户点击“手动巩固”可能与另一个请求完成写入交错，导致 pending、索引和磁盘保存顺序不一致。

## 修复方向

把 Memento 写操作统一串行化；可先粗粒度用同一把锁包住 `ingest_turn()` 和 `_consolidate()`，后续再拆成任务队列。

## 建议动作

`cs-issue`，因为这是并发一致性风险，需要复现和边界定义。
