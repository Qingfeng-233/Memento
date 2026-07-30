# Memento Memory Benchmark

这个目录放统一记忆命中评测主线。当前阶段只做主干判断，不计算 Hit@1、MRR 等量化指标。

## 评测口径

- Memento: 当前本地代码，默认使用本地 `models/Qwen3-Embedding-0.6B`，可启用 concept graph。
- Letta: Docker server + Postgres/pgvector，通过 HTTP API 访问，不依赖本地 `letta` Python 包。
- Mem0: 本地 Python + Qdrant，可选辅助 baseline。

## Letta Docker

先启动 Letta:

```powershell
docker compose -f scripts/test/letta-compose.yaml up -d
```

Letta key 路由遵循项目根目录 `AGENTS.md`:

- 容器全局 `OPENAI_API_KEY=${SILICONFLOW_API_KEY}` 用于 embedding。
- OpenCode LLM key 通过 Letta BYOK provider 注册。
- 默认 embedding: SiliconFlow `Qwen/Qwen3-Embedding-4B`。

## 运行

默认跑 Memento + Letta:

```powershell
python benchmark/compare_memory_systems.py
```

指定系统:

```powershell
python benchmark/compare_memory_systems.py --systems memento,letta,mem0
```

只快速抽样:

```powershell
python benchmark/compare_memory_systems.py --limit 20 --systems memento,letta
```

输出目录:

```text
benchmark/results/
```

每次运行会写出一份 `.md` 和一份 `.json`。
