# Memento Chat

带长期记忆的本地对话 Web 应用。用于**实测 Memento 记忆系统效果**。

每轮对话会自动：检索相关过往记忆 → 注入 prompt → 调 LLM → 回复 → 把这轮 Q/A 存入记忆库。
闲置一段时间后自动巩固（建索引 + 睡眠巩固）。

## 快速开始

```powershell
# 1. 安装服务依赖（只需一次）
pip install -r requirements-server.txt

# 2. mock 模式先跑通（不调 LLM，验证记忆闭环）
python -m apps.chat.app --mock --port 8080
# 浏览器打开 http://127.0.0.1:8080

# 3. 接真 LLM 实测（复用 .env 里的 OPENCODE_API_* / SILICONFLOW_API_*）
#    在 .env 里加一行模型名：
#       CHAT_LLM_MODEL=deepseek-ai/DeepSeek-V3.2
python -m apps.chat.app --port 8080
```

Windows 用户也可以直接运行根目录的 `start_chat.bat`，默认启动 mock 模式并自动打开浏览器。

如果要让别人访问同一个服务实例，建议启用 demo token：

```powershell
python -m apps.chat.app --mock --host 0.0.0.0 --port 8080 --token "change-me"
```

开启 token 后，网页会在首次 401 时提示输入，并把 token 保存到浏览器 localStorage。

## LLM 配置

配置从 `.env` 读取，优先级（从高到低）：

| 变量 | 说明 |
|---|---|
| `CHAT_LLM_API_BASE` | API 地址（不设则复用 `OPENCODE_API_BASE` → `SILICONFLOW_API_BASE`）|
| `CHAT_LLM_API_KEY` | API key（不设则同上链式复用）|
| `CHAT_LLM_MODEL` | 模型名，如 `deepseek-ai/DeepSeek-V3.2`。**必填**才能接真 LLM |
| `CHAT_LLM_MOCK` | `1` 强制 mock 模式 |

> `--mock` 和 `CHAT_LLM_MOCK=1` 会强制 mock，即使本地已经在网页设置里保存过 API key，也不会调用真实 LLM。
> 任何 OpenAI 兼容端点都能用。也可命令行 `--model <名>` 临时覆盖模型名。

网页设置会把 API Base / Key / Model 保存到本机 `data/chat_llm_settings.json`。该文件已加入 `.gitignore`，不要提交自己的 key。

## 命令行参数

```
--host           监听地址（默认 127.0.0.1）
--port           端口（默认 8080）
--store          Memento 存储目录（默认 data/memento_store）
--embedding-model embedding 后端（默认 tfidf-svd，零模型下载）
--model          LLM 模型名（不设则 mock）
--mock           强制 mock 模式
--retrieve-k     每轮注入的记忆条数（默认 5）
--idle-minutes   闲置多少分钟自动巩固（默认 10）
--concepts       闲置巩固时构建概念图（需 keyatten 模型，较慢）
--history-turns  拼进 prompt 的历史对话轮数（默认 6）
--token          启用 HTTP API token 保护（也可用 MEMENTO_CHAT_TOKEN）
```

## 记忆闭环怎么跑

```
用户输入
  ↓ memento.query() 检索相关记忆（记忆库 ≥3 条才注入，避免早期噪音）
  ↓ 拼进 system prompt【相关记忆】
  ↓ 调 LLM 生成回复
  ↓ memento.add_node() 把这轮 Q/A 存入（进 pending 缓冲）
  ↓ 重置闲置计时器
  ↓ 闲置 N 分钟无对话
  → build_index（pending 入索引）+ trigger_sleep（睡眠巩固）
```

实测建议：**先攒 10-20 轮对话，手动点几次"手动巩固"把记忆入索引**，
之后新对话就会自动带上相关过往记忆。每条 AI 回复上方的蓝色卡片会显示
本轮调了哪些记忆、相关性多少——这是 Memento 的可解释性体现。

侧栏“调试观测”会显示当前 session 的 turn、memory、巩固 job、retrieval 和 error 摘要，
用于判断记忆是否写入、是否入索引、是否被检索或被 retry/edit 标记为 superseded。

## 多人测试边界

- 浏览器会自动创建匿名 `session_id` 并保存到 localStorage。
- 每轮对话都会绑定 `turn_id`，写入的记忆会绑定 `session_id`、`owner_id` 和 `visibility:private`。
- history、检索注入、最近记忆列表、调试观测和删除记忆都按当前 session/owner 过滤。
- retry/edit 不再物理删除旧记录，而是把旧 turn 和旧 memory 标记为 `superseded`。
- 这仍是 demo 级匿名多人测试，不是完整账号系统；公网长期服务应再接入真实身份和部署层访问控制。

## 文件

- `app.py` — FastAPI 后端：对话循环 + 闲置巩固后台线程
- `llm.py` — LLM 调用封装（OpenAI 兼容 + mock）
- `ingest.py` — 对话→记忆切分（当前粗切：一整条 Q/A 一个节点）
- `store.py` — 聊天 demo 的 JSON 持久化抽象（session/history/job）
- `static/index.html` — 前端：对话流 + 记忆可视化 + 侧栏状态

## 开源测试注意事项

- 默认存储目录是 `data/memento_store`，这是本地运行状态，不应提交。
- 如需给测试者提供样例数据，请使用脱敏数据，不要提交个人真实对话。
- 非本机访问时务必使用 `--token` 或 `MEMENTO_CHAT_TOKEN`。
- 推荐测试流程：先用 mock 模式发 3-5 轮消息，点击“手动巩固”，再问一个相关问题，看回复上方是否出现“本轮调用了 N 条长期记忆”。
