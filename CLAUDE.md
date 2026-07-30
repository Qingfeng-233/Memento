# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

Memento 是一个面向个人长程记忆的可解释语义检索实验项目，当前版本定位为 `v0.1.0-alpha` / `preview`：核心机制已能在真实 Q/A 记忆上工作，事件拆分、情景共现和稳定排序仍在演进。

## 常见命令

环境假设：Windows 11，PowerShell，项目根目录为 `D:/工作区/项目/Memento`。

### 环境安装

```powershell
cd "D:/工作区/项目/Memento"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

核心依赖：`jieba`、`scikit-learn`、`faiss-cpu`、`numpy`、`keyatten>=0.3.1`。若使用 `torch` embedding 后端（Qwen3），需要额外安装对应版本，但不在 `requirements.txt` 中锁定。

### 运行演示 / 样例脚本

- 完整演示（TF-IDF+SVD，构建共现图、查询、睡眠、保存）：
  ```powershell
  python run_demo.py
  ```
- 对比展示（搜索 / 召回 / 项目状态）：
  ```powershell
  python show_full.py
  ```
- 完整样例（详细输出每条记忆和 RAG vs 扩散差异）：
  ```powershell
  python show_samples.py
  ```
- 这些脚本都期望根目录下存在 `memories.jsonl`。如果缺数据，会失败。

### 运行测试脚本

脚本都在 `scripts/test/` 下，用拷贝代码运行的方式验证子能力，没有统一的测试框架：

```powershell
# RAG vs 扩散联想 7 组对比
python scripts/test/test_compare.py

# 关键词建边测试
python scripts/test/test_keyword_edges.py

# 关键词过滤 / 惊奇度相关测试
python scripts/test/test_keyword_filter.py
python scripts/test/test_surprisal.py
python scripts/test/test_min_surprisal.py
```

> 不存在 `pytest` 或 `unittest` 套件；要跑“一个测试”，直接运行对应文件即可。

### Benchmark（对比 Memento / Letta / Mem0）

```powershell
# Memento + Letta（默认，Letta 需先启动）
python benchmark/compare_memory_systems.py

# 指定系统
python benchmark/compare_memory_systems.py --systems memento,letta,mem0

# 快速抽样 20 条
python benchmark/compare_memory_systems.py --limit 20 --systems memento,letta

# 关闭 Memento 概念图（纯 RAG+扩散）
python benchmark/compare_memory_systems.py --no-concepts
```

输出目录：`benchmark/results/`，每次生成 `{YYYYMMDD-HHMMSS}.md` 和 `.json`。

### 启动 Letta（外部 baseline）

Letta 固定通过 Docker HTTP server 使用，不依赖本地 `letta` Python 包：

```powershell
docker compose -f scripts/test/letta-compose.yaml up -d
```

Letta key 路由约定（见根目录 `AGENTS.md`）：

- 容器全局 `OPENAI_API_KEY=${SILICONFLOW_API_KEY}`，用于 embedding。
- OpenCode `deepseek-v4-flash` LLM key 通过 Letta BYOK provider 单独注册。
- 默认 embedding：`SiliconFlow Qwen/Qwen3-Embedding-4B`。

### 构建与索引关键 API（最小流程）

```python
from memento import Memento

mem = Memento(embedding_model="tfidf-svd", diffusion_hops=2)
mem.add_node(text="...")
mem.build_index()           # 必须调用后向量索引才生效
mem.build_concept_graph(...)  # 可选，构建关键词副节点图
mem.query("...")            # RAG + 图扩散
mem.query_with_concepts("...", debug=True)  # 概念图检索，带可解释输出
```

## 高层架构

Memento 不是单一向量检索，而是把记忆拆成两个互补系统：

1. **记忆 A：向量快速匹配（RAG）**
   - 位于 `memento/index/vector_index.py`。
   - 支持四种 embedding 后端：
     - `tfidf-svd`（默认，轻量，无需模型下载）
     - `qwen3` / 本地 Qwen3-Embedding 模型（`models/Qwen3-Embedding-0.6B`）
     - `sentence-transformers` 模型
     - `api:` 前缀的远程 embedding（默认 SiliconFlow，需 `SILICONFLOW_API_KEY`）
   - 4B/8B 模型会自动加 Qwen3 instruction prefix；0.6B 不加。
   - 使用 FAISS `IndexFlatIP` 做内积检索；向量均已 L2 归一化，内积即余弦相似度。

2. **记忆 B：图联想网络 + 激活扩散**
   - 图结构在 `memento/graph/memory_graph.py`。
   - 扩散引擎在 `memento/engine/diffusion.py`：先向量召回种子，再沿边多跳传播能量，最终按重要性/生命力加权排序，并副作用强化命中路径。
   - 衰减引擎 `memento/engine/decay.py`：管理生命力 λ、重要性 ω、边权 w 的衰减/修剪/保护，以系统时钟步推进，不依赖真实时间。
   - 睡眠引擎 `memento/engine/sleep.py`：离线回放巩固、随机游走、向量探索建边、贪心聚类凝聚、全局遗忘修剪。

3. **关键词副节点图（可解释检索核心）**
   - 关键词作为独立节点，构建在 `memento/concept/concept_graph.py`。
   - `build_concept_graph()` 提取关键词后：
     - 关键词 ↔ 记忆：根据关键词初始能量建边；
     - 关键词 ↔ 关键词：根据向量余弦相似度建边，并用 `exponential_edge_weight` 指数化拉开强弱差距。
   - `query_with_concepts()` 通过查询直接激活关键词、扩散、再聚合回记忆得分；`debug=True` 可查看 `seed_concepts` / `activated_concepts` / `concept_supports`。

4. **主 API 封装**
   - `memento/api.py` 中的 `Memento` 类负责把上述模块串起来，提供 `add_node`、`build_index`、`activate`、`build_keyword_edges`、`build_concept_graph`、`query`、`query_with_concepts`、`trigger_sleep`、`save/load` 等接口。
   - `add_node` 在 `build_index()` 前只是把节点放进 `_pending_nodes` 缓冲区；必须调用 `build_index()` 才会真正编码向量并加入索引和图。
   - `save/load` 会把节点、边、FAISS 索引、ID 映射、关键词 IDF 都落盘到指定目录。

## 关键约定

- **默认本地模型路径**：`models/Qwen3-Embedding-0.6B`。该路径在 `README.md`、`benchmark/compare_memory_systems.py`、`memento/api.py` 多处以相对根目录的字符串形式出现。
- **大模型/Embedding Key 路由**：
  - SiliconFlow embedding：`SILICONFLOW_API_BASE`、`SILICONFLOW_API_KEY`。
  - OpenCode LLM：`OPENCODE_API_BASE`、`OPENCODE_API_KEY`。
  - benchmark 脚本会从根目录 `.env` 加载环境变量。
- **Letta 评测约定**：只走 Docker HTTP API，不依赖本地 `letta` Python 包；统一适配器模式为 `add_memory / search / cleanup`。
- **Mem0 评测约定**：本地 Python + Qdrant，作为可选辅助 baseline。
- **当前限制**：事件拆分尚未实现；情景共现 `activate()` 可建边但完整的长期情景结构未成型；抽象查询 Top 1 排序可能漂移；概念图构建较慢（145 条 Q/A 约 150s），依赖 `keyatten` 缓存优化。
- **许可证**：仓库尚未声明 `LICENSE`，发布到公开平台前应当补充。
