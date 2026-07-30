# Memento

Memento 是一个面向个人长程记忆的可解释语义检索实验项目。

当前版本建议定位为 `v0.1.0-alpha` / `preview`：核心机制已经能在真实 Q/A 记忆上工作，但事件拆分、情景共现和稳定排序策略还在演进中。

## 最快开始：本地 Q/A 服务

如果只需要“写入一条问答，再按自然语言查询”，无需理解图、索引或模型配置。以下服务默认使用轻量 `tfidf-svd` 后端，不下载模型、不需要 API Key，所有数据只保存在本机的单个 SQLite 文件 `data/memento.db`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-server.txt
python -m memento.local_service
```

服务默认监听 `http://127.0.0.1:8765`。另开一个 PowerShell 窗口执行：

```powershell
Invoke-RestMethod "http://127.0.0.1:8765/memories" -Method Post `
  -ContentType "application/json" `
  -Body (@{ question = "服务默认监听哪个端口？"; answer = "默认监听 127.0.0.1:8765。" } | ConvertTo-Json)

Invoke-RestMethod "http://127.0.0.1:8765/search" -Method Post `
  -ContentType "application/json" `
  -Body (@{ query = "本地服务的默认端口是什么？"; limit = 5 } | ConvertTo-Json)
```

接口只有三个：`POST /memories` 写入 Q/A，`POST /search` 查询，`GET /health` 查看已保存条数。写入成功后立即可查询；服务重启后会从同一数据库文件自动恢复。

> `data/memento.db` 与 `data/testtxt.txt` 都是本地数据，已被 `.gitignore` 排除。请勿将本地数据库或未脱敏评测数据提交到公开仓库。

## 核心思路

Memento 不只做一次向量相似度检索，而是把记忆拆成两层：

- **主节点**：一条完整记忆，当前通常是一组 Q/A 或一段文本。
- **副节点**：由 KeyAtten 抽取出的关键词 / 短语概念。

检索时先用向量召回候选，再通过关键词副节点做语义扩散，把字面不同但概念相近的记忆拉入候选。例如：

- `容器部署`
- `Docker Compose`
- `服务编排`
- `健康检查`

这些词不一定字面重合，但在概念图里可以通过向量相似度、关键词能量和指数化边权建立连接。

## 当前能力

- 向量索引检索。
- 记忆图扩散检索。
- KeyAtten 3.1 关键词抽取。
- 关键词副节点概念图。
- 基于 IDF / 初始能量的泛词过滤。
- 关键词之间的向量语义边。
- 指数化边权，用于拉开相近词和弱相关词的差距。
- NLP 短语合并，减少中间词、泛词污染。
- Debug 模式，可查看 seed concepts、activated concepts 和每条结果的概念支持。
- Benchmark 主线，可对比 Memento、Letta、Mem0。

## 安装

建议使用独立虚拟环境。

```powershell
cd <Memento 项目目录>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

默认依赖只包含核心路径：

```text
jieba
scikit-learn
faiss-cpu
numpy
keyatten>=0.3.1
```

如果使用 KeyAtten + Qwen3 Embedding，建议准备本地模型：

```text
models/Qwen3-Embedding-0.6B
```

当前默认关键词模型路径就是：

```python
"models/Qwen3-Embedding-0.6B"
```

## 最小示例

```python
from memento import Memento

mem = Memento(
    embedding_model="tfidf-svd",
    diffusion_hops=2,
)

mem.add_node(
    "用户问: Docker 容器启动后立即退出怎么排查？\n"
    "回答: 先查看容器日志，再检查启动命令和必需的环境变量。"
)
mem.add_node(
    "用户问: Compose 服务怎样加载环境变量？\n"
    "回答: 可以使用 environment 或 env_file 配置服务所需变量。"
)
mem.add_node(
    "用户问: PostgreSQL 数据库怎样创建备份？\n"
    "回答: 可以使用 pg_dump 导出，并定期验证备份能否恢复。"
)

mem.build_index()

results = mem.query("容器为什么启动失败", k=3)
for item in results:
    print(item["score"], item["text"][:80])
```

## 概念图检索示例

概念图是当前版本最值得关注的能力。它会把关键词作为独立副节点，并通过语义相似度建立关键词之间的边。

```python
from memento import Memento

mem = Memento(embedding_model="tfidf-svd")

texts = [
    "用户问: Docker 容器启动后立即退出，应该先检查什么？回答: 先查看容器日志和启动命令。",
    "用户问: Compose 服务如何注入配置？回答: 可以通过 env_file 加载环境变量。",
    "用户问: 应用启动时报缺少配置项怎么办？回答: 检查部署环境是否提供了必需变量。",
    "用户问: PostgreSQL 如何做定期备份？回答: 使用 pg_dump 并验证恢复流程。",
]

for i, text in enumerate(texts):
    mem.add_node(text, node_id=f"mem_{i:03d}")

mem.build_index()

info = mem.build_concept_graph(
    top_k=8,
    keyword_method="keyatten",
    keyword_model="models/Qwen3-Embedding-0.6B",
    keyword_device="cuda",
    keyword_dtype="float16",
    keyword_cache_enabled=True,
    keyword_cache_dir="data/keyatten_cache",
    max_concepts=500,
    min_concept_energy=0.5,
    keyword_sim_threshold=0.45,
    keyword_temperature=0.06,
)

print(info)

debug = mem.query_with_concepts(
    "容器启动时配置丢失怎么排查",
    k=5,
    concept_weight=0.35,
    debug=True,
)

print("Seed concepts:")
for concept in debug["seed_concepts"]:
    print(concept)

print("Results:")
for item in debug["results"]:
    print(item["score"], item["rag_score"], item["concept_score"], item["text"][:100])
    print("supports:", item.get("concept_supports", [])[:3])
```

## 手工连接概念

如果你明确知道两个概念应该相连，可以手工加边：

```python
mem.link_concepts("容器启动", "环境变量", weight=0.9)
mem.link_concepts("环境变量", "服务配置", weight=0.7)
```

这适合用于少量高置信的人工校正，不建议把它当成主要数据来源。

## Benchmark

评测主线在 `benchmark/`。

默认评测 Memento + Letta：

```powershell
python benchmark/compare_memory_systems.py
```

指定系统：

```powershell
python benchmark/compare_memory_systems.py --systems memento,letta,mem0
```

快速抽样：

```powershell
python benchmark/compare_memory_systems.py --limit 20 --systems memento,letta
```

输出目录：

```text
benchmark/results/
```

当前评测口径是主干观察，不计算 Hit@1、MRR 等量化指标。

最近一次完整评测：

```text
benchmark/results/20260613-202248.md
benchmark/results/20260613-202248.json
```

观察结论：

- Memento 在工具型、事实型、主题明确的查询上表现稳定。
- Memento 已经能把 `容器启动`、`环境变量`、`服务配置` 这类相关概念拉入候选。
- Letta 在抽象自然语言意图理解上仍更强。
- Memento 的优势是路径更可解释、可控，适合继续工程化调参。

## Letta 对比基线

Letta 使用 Docker HTTP server，不依赖本地 Python `letta` 包。

启动：

```powershell
docker compose -f scripts/test/letta-compose.yaml up -d
```

项目约定：

- Letta 作为主要外部 baseline。
- Mem0 作为可选辅助 baseline。
- Letta key routing 见根目录 `AGENTS.md`。

## Debug 输出

`query_with_concepts(..., debug=True)` 会返回：

- `seed_concepts`：查询直接激活的关键词概念。
- `activated_concepts`：扩散后的高激活概念。
- `results[].concept_supports`：每条结果由哪些概念支持。

这部分是 Memento 相比黑盒记忆系统的重要差异：你可以看到一条记忆为什么被拉出来。


## 许可证

本项目采用 [MIT License](LICENSE)。
