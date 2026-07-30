# Memento 项目约定

## 环境与依赖约定

- 项目使用独立 Python 虚拟环境，根目录 `requirements.txt` 只锁定核心路径依赖（`jieba`、`scikit-learn`、`faiss-cpu`、`numpy`、`keyatten>=0.3.1`）。
- 若使用 Qwen3 Embedding 或 keyatten，需要本地模型目录 `models/Qwen3-Embedding-0.6B`；这是当前默认关键词模型路径，也以字符串形式硬编码在 `memento/api.py` 和 `benchmark/compare_memory_systems.py` 中。
- embedding 后端通过 `Memento(embedding_model=...)` 选择：
  - `"tfidf-svd"`（默认，无需下载模型）
  - 本地 `"models/Qwen3-Embedding-0.6B"`
  - `"sentence-transformers"` 兼容模型名
  - `"api:Qwen/Qwen3-Embedding-4B"` 等远程 API（需 `SILICONFLOW_API_KEY`）
- 环境变量读取：benchmark 脚本会从根目录 `.env` 加载；远程 API 需要 `SILICONFLOW_API_BASE`、`SILICONFLOW_API_KEY`、`OPENCODE_API_BASE`、`OPENCODE_API_KEY`。
- 服务可选依赖独立在 `requirements-server.txt`（`fastapi`、`uvicorn`、`mcp`、`pydantic`），核心 `requirements.txt` 不受污染。仅启动 HTTP/MCP 服务时安装：`pip install -r requirements-server.txt`。

## 核心 API 使用约定

- `add_node()` 只是暂存到缓冲区，必须调用 `build_index()` 后向量索引和图才生效。
- `build_concept_graph()` 依赖 `build_index()` 已完成。
- `query_with_concepts()` 依赖 `build_concept_graph()` 已完成；`debug=True` 会返回 `seed_concepts`、`activated_concepts`、`results[].concept_supports`。
- keyatten 内部缓存默认开启，目录为 `data/keyatten_cache`；不要把这个目录提交到 git（已在 `.gitignore` 中）。

## 对外接口（CLI / HTTP / MCP）

三端共用 `memento/store.py` 的存储助手，默认存储目录 `data/memento_store/`（可用 `--store` 或环境变量 `MEMENTO_STORE` 覆盖）。embedding 后端通过 `--embedding-model` 或 `MEMENTO_EMBEDDING_MODEL` 指定，默认 `tfidf-svd`；store_meta.json 会记录所用后端，下次加载自动对齐，避免 FAISS 维度不匹配。

- **CLI**（`memento/cli.py`，纯 stdlib，无额外依赖）：`python -m memento.cli <command>`。一次性进程模型——每次从磁盘 load，写操作后自动 save（`--no-save` 关闭）。子命令覆盖 add/import/build/query/query-concepts/query-rag/get/stats/link/link-concepts/activate/mark-important/sleep/clock/save，外加 `serve`、`mcp` 两个转发子命令。读命令用 `--json` 输出机器可读结果。
- **HTTP**（`memento/server.py`，需 `requirements-server.txt`）：`python -m memento.cli serve` 或 `python -m memento.server`。FastAPI，常驻内存，启动时 load、写操作后自动落盘（`--no-autosave` 关闭后用 `POST /save` 手动控制）。端点：`/add`、`/import`、`/build-index`、`/build-concept-graph`、`/build-keyword-edges`、`/query`、`/query-concepts`、`/query-rag`、`/activate`、`/link`、`/link-concepts`、`/mark-important`、`/clock-step`、`/sleep`、`/save`、`/load`、`GET /node/{id}`、`GET /stats`，根路径 `/` 返回端点清单。
- **MCP**（`memento/mcp_server.py`，需 `requirements-server.txt`）：`python -m memento.cli mcp` 或 `python -m memento.mcp_server`。stdio transport，暴露完整 API 共 21 个工具。Memento 实例延迟到首次工具调用才加载，避免启动阻塞客户端。

存储文件（`Memento.save()` 产物）：`nodes.json`、`edges.json`、`vectors.faiss`、`id_map.json`、`meta.json`（含 `clock_step`/`index_built`/`pending_nodes`）、`keywords.json`、`tfidf_pipeline.pkl`（tfidf-svd 后端的拟合管道，load 后 encode 新查询需要）、概念图文件、`store_meta.json`（embedding 后端记录）。`vectors.faiss` 与 `tfidf_pipeline.pkl` 在索引未构建时不存在，`load()` 已处理该情况。

## 评测与测试约定

- 测试脚本位于 `scripts/test/`，没有统一测试框架；验证子能力直接运行对应文件即可，例如 `python scripts/test/test_compare.py`。
- benchmark 主线在 `benchmark/compare_memory_systems.py`。
- benchmark 默认评测 Memento + Letta；运行含 Letta 的 benchmark 前必须先启动 Docker：`docker compose -f scripts/test/letta-compose.yaml up -d`。
- benchmark 数据默认使用 `data/testtxt.txt`；数据格式为 `【用户提问】...【AI 回答】...` 分隔的 Q/A 对。
- **记忆可用性评测指标**：对于缺乏标准答案的评测集，采用“记忆可用性”（即检索到的记忆是否足以回答所查询到的问题）作为核心评估指标。详见设计文档 [docs/26-6-22-memory-usability-eval.md](file:///D:/工作区/项目/Memento/docs/26-6-22-memory-usability-eval.md)。

## Letta 评测方式

Letta 作为 Memento 的主 baseline 时，固定使用 Docker server，不依赖本地 Python `letta` 包或本地 Python 依赖环境。

推荐口径：

- Memento: 当前本地代码。
- Mem0: 本地 Python + Qdrant，作为辅助成熟库 baseline。
- Letta: Docker server + Postgres/pgvector，通过 HTTP API 访问。

相关文件：

- `scripts/test/letta-compose.yaml`
- `scripts/test/verify_letta_memory.py`
- `scripts/test/evaluate_letta_memory.py`

Letta key 路由约定：

- Letta 容器全局 `OPENAI_API_KEY=${SILICONFLOW_API_KEY}`，用于 OpenAI-compatible embedding client。
- OpenCode `deepseek-v4-flash` LLM key 通过 Letta BYOK provider 单独注册。
- 默认 embedding: SiliconFlow `Qwen/Qwen3-Embedding-4B`。

后续统一对比脚本应把 Letta 当作 HTTP adapter：

```python
class LettaAdapter:
    add_memory(text, metadata)
    search(query, top_k)
    cleanup()
```

不要在统一评测中直接依赖本地 `letta` Python 包，避免与 spaCy、transformers、typer 等 NLP/模型依赖互相污染。
