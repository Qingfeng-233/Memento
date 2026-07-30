---
doc_type: audit-index
audit: 2026-07-08-chat-demo-memory
scope: apps/chat 聊天 demo 与 Memento 记忆闭环的多人/多轮使用路径
created: 2026-07-08
status: active
total_findings: 6
---

# chat-demo-memory 审计报告

## 范围

本次只扫描 `apps/chat/` 对话 demo、`memento/api.py` 的节点写入路径，以及 `memento/index/vector_index.py` 的索引映射路径。测试方式以临时 store + FastAPI TestClient 为主，模拟 Alice / Bob 交替对话、巩固、重试、删除后继续对话。

## 总评

聊天 demo 的单人最小闭环是通的：消息能进入 pending，手动巩固后可入索引，新问题能拿到 `memories_used`。但它现在不是多人安全模型，核心状态是全局单例，历史、检索、重试、删除都没有 user/session/turn 归属。最值得优先处理的是：全局历史串 prompt、跨用户检索泄露、`replace_last` 删除错轮次、删除后 node_id 碰撞导致新记忆丢失。

## 发现清单

| # | 性质 | 严重度 | 置信度 | 标题 | 文件 |
|---|---|---|---|---|---|
| 1 | security | P1 | high | 全局 history / memory 导致多人对话互相泄露 | [finding-01.md](finding-01.md) |
| 2 | bug | P1 | high | `replace_last` 按全局最后一轮回滚，会删错用户和错节点 | [finding-02.md](finding-02.md) |
| 3 | bug | P1 | high | 删除节点后按数量生成 ID 会撞已有节点，新记忆静默丢失 | [finding-03.md](finding-03.md) |
| 4 | security | P1 | medium | 缺少鉴权/访问边界，读写和清空接口可被任意本地 HTTP 调用 | [finding-04.md](finding-04.md) |
| 5 | security | P2 | high | 长期记忆存储了原始 assistant 回复，包含 `<think>` 思考段 | [finding-05.md](finding-05.md) |
| 6 | bug | P2 | medium | 巩固过程没有持有状态锁，可能与聊天写入并发修改同一 Memento | [finding-06.md](finding-06.md) |

## 按维度分布

| 性质 | P0 | P1 | P2 | 合计 |
|---|---|---|---|---|
| bug | 0 | 2 | 1 | 3 |
| security | 0 | 2 | 1 | 3 |
| performance | 0 | 0 | 0 | 0 |
| maintainability | 0 | 0 | 0 | 0 |
| arch-drift | 0 | 0 | 0 | 0 |
| **合计** | **0** | **4** | **2** | **6** |

## 下一步建议

- **P1 本迭代修**：Finding 1、2、3、4。大改前建议先决定 demo 是“单用户本地工具”还是“多人测试服务”；如果是后者，必须先引入 session/user 边界。
- **P2 有空再看**：Finding 5、6。它们不会阻断单人 demo，但会污染长期记忆质量，并在并发或长任务下产生难复现问题。
