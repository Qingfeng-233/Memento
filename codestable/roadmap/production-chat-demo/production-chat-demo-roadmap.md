---
doc_type: roadmap
slug: production-chat-demo
status: active
created: 2026-07-08
last_reviewed: 2026-07-08
tags: [chat-demo, productionization, memory, session, security]
related_requirements: []
related_architecture: []
---

# Production Chat Demo Roadmap

## 1. 背景

当前 `apps/chat` 已经能证明单人本地长期记忆闭环：用户输入、检索相关记忆、调用 LLM、保存 Q/A、手动或闲置巩固后进入索引。但审计显示它仍是单进程单用户 demo：history、memory、retry/delete 都是全局状态，缺少 user/session/turn 归属；删除节点后还会出现 ID 碰撞并导致新记忆静默丢失。

本 roadmap 的目标是把聊天 demo 推到“可公开测试 / 准生产”的基础状态。这里的生产级不是先做复杂账号、分布式部署或企业权限，而是先把状态模型、数据一致性、安全边界和可观测性补齐，让后续检索算法、概念图、记忆切分的大改不会被基础串扰污染判断。

## 2. 范围与明确不做

### 本 roadmap 覆盖

- 稳定记忆节点 ID，避免删除后继续新增导致 ID 碰撞。
- 引入 `session_id` / `turn_id` / `owner_id` 的状态归属模型。
- 让 history、memory、retrieval、edit/retry/delete 按归属边界工作。
- 建立对话入库清洗管线，避免 `<think>`、流式错误、调试文本进入长期记忆。
- 把巩固过程抽象为串行 job，避免请求线程并发修改同一 Memento。
- 给本地 demo 增加最小安全边界，保护读写、删除、清空、设置类接口。
- 为后续 SQLite/Postgres 持久化抽象 store 接口。
- 增加基础可观测性：turn 状态、memory 写入状态、巩固 job、检索解释和错误日志。

### 明确不做

- 不做完整账号注册 / 登录 / OAuth / SSO。第一阶段只做本地 token 和匿名 session。
- 不做分布式部署、横向扩容和多 worker 共享内存问题。
- 不重写 Memento 核心检索算法；本 roadmap 只约束聊天 demo 如何调用和隔离它。
- 不引入复杂前端框架；现有单页 HTML 可以先继续承载 demo。
- 不做企业级权限模型；只做 owner/session/visibility 三层最小归属。

## 3. 模块拆分（概设）

```text
production-chat-demo
├── identity：管理匿名 session、本地 demo token、owner 边界
├── conversation：管理 turn、history、retry/edit 状态机
├── memory-write：管理对话入库、清洗、node_id、memory metadata
├── retrieval：按 owner/session/visibility 执行检索并返回解释卡片
├── consolidation：串行巩固任务、job 状态、日志
├── persistence：抽象文件 store，为 SQLite/Postgres 迁移留接口
├── security：鉴权、危险操作保护、key 管理边界
└── observability：turn/job/error/retrieval 调试面板和事件日志
```

### identity · 身份与会话边界

- **职责**：创建和解析 `session_id`，维护匿名 `owner_id`，校验本地 demo token。它不负责完整账号体系。
- **承载的子 feature**：`session-isolation`, `demo-auth-token`, `multi-user-ready-mode`
- **触碰的现有代码 / 模块**：`apps/chat/app.py`、前端 `static/index.html`，新增轻量 session 存储助手。

### conversation · 对话轮次模型

- **职责**：把每轮对话建模为 `ChatTurn`，管理 streaming/done/failed/superseded 状态，支持按 turn 精确 retry/edit。它不直接操作 Memento 索引。
- **承载的子 feature**：`turn-model`, `safe-retry-edit`
- **触碰的现有代码 / 模块**：`apps/chat/app.py`、`apps/chat/static/index.html`、`apps/chat/ingest.py`

### memory-write · 记忆写入管线

- **职责**：把完成的 `ChatTurn` 转换成 `MemoryRecord`，生成稳定 `node_id`，清洗 assistant 回复，写入 Memento pending。它不负责检索排序。
- **承载的子 feature**：`stable-memory-ids`, `memory-ingest-cleaner`
- **触碰的现有代码 / 模块**：`memento/api.py`、`apps/chat/ingest.py`、`memento/store.py`

### retrieval · 隔离检索与解释卡片

- **职责**：按 `owner_id/session_id/visibility` 过滤候选，调用 Memento 检索，并把 `memories_used` 返回给前端。它不写入记忆。
- **承载的子 feature**：`session-isolation`, `multi-user-ready-mode`, `observability-panel`
- **触碰的现有代码 / 模块**：`apps/chat/app.py`、`memento/api.py` 的查询结果包装层。

### consolidation · 巩固任务

- **职责**：把手动/闲置巩固封装为串行 job，记录状态、开始/结束时间、错误、步骤日志。它是唯一允许执行 `build_index()` / `trigger_sleep()` 的后台入口。
- **承载的子 feature**：`consolidation-queue`, `observability-panel`
- **触碰的现有代码 / 模块**：`apps/chat/app.py` 中 `_consolidate()` / `_idle_watcher()`。

### persistence · 持久化抽象

- **职责**：定义 ChatSession、ChatTurn、MemoryRecord、LLM settings、consolidation job 的读写接口。第一阶段可落到 JSON 文件，后续替换 SQLite/Postgres。
- **承载的子 feature**：`store-abstraction-v2`
- **触碰的现有代码 / 模块**：`memento/store.py`、`apps/chat/app.py`、新增 `apps/chat/store.py`。

### security · 安全边界

- **职责**：保护设置、历史、记忆、删除、清空等接口；管理本地 API token 和危险操作二次确认。它不做完整用户系统。
- **承载的子 feature**：`demo-auth-token`, `multi-user-ready-mode`
- **触碰的现有代码 / 模块**：`apps/chat/app.py`、`apps/chat/static/index.html`。

### observability · 可观测性

- **职责**：统一展示最近 turn、memory 写入、巩固 job、检索解释、错误日志。它不改变核心行为。
- **承载的子 feature**：`observability-panel`
- **触碰的现有代码 / 模块**：`apps/chat/app.py`、`apps/chat/static/index.html`。

## 4. 模块间接口契约 / 共享协议（架构层详设）

### 4.1 会话创建协议

**方向**：前端 → identity
**形式**：HTTP API

```text
POST /api/session
Request:  { client_label?: str }
Response: {
  session_id: str,
  owner_id: str | null,
  created_at: str
}
错误：400 invalid_input, 500 internal
```

**约束**：

- `session_id` 必须由后端生成，推荐 UUID/ULID。
- 前端必须把 `session_id` 存在浏览器 localStorage。
- 没有完整账号体系前，`owner_id` 可以等于匿名设备级 owner，或为 `null`；但所有 ChatTurn 和 MemoryRecord 必须带 `session_id`。

### 4.2 对话流协议

**方向**：前端 → conversation → retrieval / memory-write
**形式**：HTTP SSE

```text
POST /api/chat
Request: {
  session_id: str,
  text: str,
  retry_of_turn_id?: str,
  edit_of_turn_id?: str
}

SSE event data:
info: {
  type: "info",
  turn_id: str,
  memories_used: MemoryCard[],
  mock: bool
}
content: {
  type: "content",
  turn_id: str,
  content: str
}
error: {
  type: "error",
  turn_id: str | null,
  error: str,
  code?: str
}
done: {
  type: "done",
  turn_id: str,
  node_id: str | null
}
```

**约束**：

- 每次请求必须创建一个新的 `turn_id`。
- `retry_of_turn_id` / `edit_of_turn_id` 只能指向同一 `session_id` 下已有 turn。
- 老 turn 不物理删除，标记为 `superseded`，并在 memory 层标记旧 node 不再参与检索。
- `node_id` 只有在记忆写入成功后返回；写入失败必须返回 `node_id: null` 和可观测错误。

### 4.3 ChatTurn 数据结构

**方向**：conversation ↔ persistence
**形式**：共享数据结构

```python
class ChatTurn:
    id: str
    session_id: str
    owner_id: str | None
    user_text: str
    assistant_text: str
    memory_node_id: str | None
    status: Literal["streaming", "done", "failed", "superseded"]
    supersedes_turn_id: str | None
    created_at: str
    updated_at: str
```

**约束**：

- `status="done"` 的 turn 才能进入长期记忆。
- `status="superseded"` 的 turn 不参与 prompt history。
- prompt history 必须按 `session_id` 过滤。

### 4.4 MemoryRecord 数据结构

**方向**：memory-write / retrieval / persistence
**形式**：共享数据结构 + Memento node metadata

```python
class MemoryRecord:
    node_id: str
    owner_id: str | None
    session_id: str
    turn_id: str
    text: str
    visibility: Literal["private", "session", "public"]
    source: Literal["chat"]
    status: Literal["active", "superseded", "deleted"]
    created_at: str
```

**约束**：

- `node_id` 必须稳定唯一，不得基于当前 node 数量生成。
- `visibility="private"` 只能被同 owner 检索。
- `visibility="session"` 只能被同 session 检索。
- `status!="active"` 的记录不得进入检索结果。

### 4.5 检索卡片协议

**方向**：retrieval → 前端
**形式**：HTTP/SSE payload

```python
class MemoryCard:
    node_id: str
    turn_id: str | None
    score: float
    text: str
    source: str
    created_at: str | None
```

**约束**：

- `MemoryCard` 只能包含当前 session/owner 可见的记忆。
- 前端展示的记忆卡片必须使用 `node_id`，不能再使用不稳定的列表序号操作删除。

### 4.6 巩固 job 协议

**方向**：conversation / UI → consolidation
**形式**：HTTP API + 后台队列

```text
POST /api/consolidate
Request: { session_id?: str }
Response: { job_id: str, status: "queued" | "running" | "done" | "failed" }

GET /api/consolidation-jobs?session_id=...
Response: ConsolidationJob[]
```

```python
class ConsolidationJob:
    id: str
    session_id: str | None
    status: Literal["queued", "running", "done", "failed"]
    steps: list[str]
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
```

**约束**：

- 同一 store 同一时间只能有一个巩固 job 修改 Memento。
- 聊天写入和巩固必须通过同一串行写锁或任务队列协调。
- job 失败不得清空 pending。

### 4.7 安全协议

**方向**：前端 / 外部客户端 → security
**形式**：HTTP header

```text
Header: X-Memento-Token: <token>
```

**受保护接口**：

- `GET /api/history`
- `GET /api/memories`
- `DELETE /api/memory/{node_id}`
- `POST /api/reset-history`
- `POST /api/reset-memory`
- `GET/POST /api/settings`
- `POST /api/models`
- `POST /api/consolidate`

**约束**：

- 默认监听 `127.0.0.1` 时允许 token 为空，但文档必须提示风险。
- 监听非 loopback host 时必须要求 token。
- `reset-memory` 必须要求 token，且前端保留二次确认。

### 4.8 Store 抽象

**方向**：各模块 → persistence
**形式**：函数接口

```python
class ChatStore:
    def create_session(self, client_label: str | None = None) -> ChatSession: ...
    def get_session(self, session_id: str) -> ChatSession | None: ...
    def append_turn(self, turn: ChatTurn) -> None: ...
    def update_turn(self, turn: ChatTurn) -> None: ...
    def list_turns(self, session_id: str, limit: int | None = None) -> list[ChatTurn]: ...
    def add_memory_record(self, record: MemoryRecord) -> None: ...
    def update_memory_status(self, node_id: str, status: str) -> None: ...
    def list_memory_records(self, session_id: str | None, owner_id: str | None, limit: int) -> list[MemoryRecord]: ...
    def append_consolidation_job(self, job: ConsolidationJob) -> None: ...
    def update_consolidation_job(self, job: ConsolidationJob) -> None: ...
```

**约束**：

- 第一阶段可以 JSON 文件实现。
- 调用方不得直接读写 `chat_history.json`。
- SQLite/Postgres 迁移时不改变上层 API。

## 5. 子 feature 清单

1. **stable-memory-ids** — 把 Memento 默认 node_id 改为稳定唯一 ID，并补删除后新增的回归测试。
   - 所属模块：memory-write
   - 依赖：无
   - 状态：done
   - 对应 feature：2026-07-08-stable-memory-ids
   - 备注：最小闭环第一步，防止记忆静默丢失。

2. **turn-model** — 引入 ChatTurn 与 turn_id，让每轮对话、历史、记忆节点建立明确绑定。
   - 所属模块：conversation
   - 依赖：`stable-memory-ids`
   - 状态：done
   - 对应 feature：2026-07-08-turn-model

3. **session-isolation** — 前端创建/保存 session_id，后端按 session 过滤 history 和 retrieval。
   - 所属模块：identity / retrieval / conversation
   - 依赖：`turn-model`
   - 状态：done
   - 对应 feature：2026-07-08-session-isolation

4. **memory-ingest-cleaner** — 入库前清洗 `<think>`、流式错误、Mock 调试文本，统一记忆格式。
   - 所属模块：memory-write
   - 依赖：`turn-model`
   - 状态：done
   - 对应 feature：2026-07-08-memory-ingest-cleaner

5. **safe-retry-edit** — 用 turn_id 精确 retry/edit，旧 turn 和旧 memory 标记 superseded。
   - 所属模块：conversation / memory-write
   - 依赖：`session-isolation`, `memory-ingest-cleaner`
   - 状态：done
   - 对应 feature：2026-07-08-safe-retry-edit

6. **consolidation-queue** — 把手动/闲置巩固改为串行 job，统一记录 job 状态与错误。
   - 所属模块：consolidation
   - 依赖：`stable-memory-ids`
   - 状态：done
   - 对应 feature：2026-07-08-consolidation-queue

7. **demo-auth-token** — 给本地 demo 增加 token 保护，危险接口和设置接口需要通过校验。
   - 所属模块：security / identity
   - 依赖：`session-isolation`
   - 状态：done
   - 对应 feature：2026-07-08-demo-auth-token

8. **store-abstraction-v2** — 新增 ChatStore 抽象，替代散落的 history/settings/job JSON 读写。
   - 所属模块：persistence
   - 依赖：`turn-model`, `session-isolation`
   - 状态：done
   - 对应 feature：2026-07-08-store-abstraction-v2

9. **observability-panel** — 前端和 API 展示 turn 状态、memory 写入、job、错误和检索解释。
   - 所属模块：observability
   - 依赖：`consolidation-queue`, `store-abstraction-v2`
   - 状态：done
   - 对应 feature：2026-07-08-observability-panel

10. **multi-user-ready-mode** — 在 owner_id / visibility / resource ownership 层补齐多人共享服务的最小边界。
    - 所属模块：identity / retrieval / security
    - 依赖：`demo-auth-token`, `store-abstraction-v2`, `safe-retry-edit`
    - 状态：done
    - 对应 feature：2026-07-08-multi-user-ready-mode

**最小闭环**：第 1 条 `stable-memory-ids` 做完后，删除中间记忆再继续对话不会静默丢失新记忆；这是后续所有记忆质量测试的最低可信前提。

## 6. 排期思路

排期按“先保证数据不丢，再保证状态不串，再保证可测可观测”的顺序推进。`stable-memory-ids` 是第一条，因为它不依赖上层产品决策，却会直接破坏长期记忆可信度。`turn-model` 和 `session-isolation` 紧随其后，因为它们定义后续编辑、检索、安全、store 抽象的归属字段。

前 4 条完成后，demo 可作为单用户本地工具可靠测试；前 7 条完成后，可以谨慎让多人使用同一个服务实例测试；全部完成后，才适合继续考虑数据库迁移、部署和更复杂的记忆算法改造。

## 7. 观察项

- 当前项目没有 `codestable/requirements/` 和 `codestable/architecture/` 现状档案；本 roadmap 没有同步修改这些目录。
- 当前 `apps/chat` 仍是未跟踪目录，发布前需要明确纳入 git 文件集。
- 已跟踪的个人测试数据和缓存不由本 roadmap 处理，开源前仍需单独清理。
- 如果后续决定“只做单用户本地工具”，`multi-user-ready-mode` 可以降级或 dropped，但 `stable-memory-ids`、`turn-model`、`memory-ingest-cleaner` 仍建议保留。

## 8. 变更日志

- 2026-07-08：创建 roadmap，基于 chat demo 多人对话审计结果拆出 10 条生产化子 feature。
- 2026-07-08：完成 `stable-memory-ids`，默认 `node_id` 改为持久化单调序列，删除后不再复用已有编号。
- 2026-07-08：完成 `turn-model`，聊天 SSE、前端 DOM turn、history 和记忆 node 建立 `turn_id` / `node_id` 绑定。
- 2026-07-08：完成 `session-isolation`，前端持久化匿名 session，后端按 session 过滤 history、记忆列表和检索注入。
- 2026-07-08：完成 `memory-ingest-cleaner`，长期记忆入库前清理 `<think>`、stream error 和 mock 调试回复。
- 2026-07-08：完成 `safe-retry-edit`，retry/edit 按 `turn_id` 标记旧 turn 和旧 memory 为 superseded，不再物理删除。
- 2026-07-08：完成 `consolidation-queue`，聊天写入、删除、清空和巩固共享 Memento 写锁，手动/闲置巩固统一记录 job。
- 2026-07-08：完成 `demo-auth-token`，配置 token 后聊天、历史、记忆、设置、巩固和危险接口要求 `X-Memento-Token`。
- 2026-07-08：完成 `store-abstraction-v2`，新增 JSON ChatStore，集中管理 session、history 和 consolidation job 持久化。
- 2026-07-08：完成 `observability-panel`，新增 `/api/observability` 和前端调试观测面板，展示 turn、memory、job、retrieval、error。
- 2026-07-08：完成 `multi-user-ready-mode`，memory 写入 owner/visibility 标签，检索、列表和删除按 session owner 可见性校验。
