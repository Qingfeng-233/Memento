"""
LLM 调用封装 — OpenAI 兼容，可配置，带 mock 模式

配置优先级（从高到低）:
  1. 显式 mock / CHAT_LLM_MOCK=1（强制不调 API）
  2. 显式传入 LLMConfig
  3. 本地设置文件 data/chat_llm_settings.json
  4. 环境变量 CHAT_LLM_API_BASE / CHAT_LLM_API_KEY / CHAT_LLM_MODEL
  5. 复用 OPENCODE_API_BASE / OPENCODE_API_KEY
  6. 复用 SILICONFLOW_API_BASE / SILICONFLOW_API_KEY
  7. mock 模式（不调 API，返回固定回复，用于先跑通记忆闭环）

.env 里已配好 OPENCODE_API_* / SILICONFLOW_API_*，默认直接复用，
只需设 CHAT_LLM_MODEL 指定模型名即可开跑。
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator


@dataclass
class LLMConfig:
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    system_prompt: str = (
        "你是一个有长期记忆的助手。下方【相关记忆】是你过往对话中沉淀的"
        "个人化知识，请在回答时自然地参考它们，不必逐条复述，除非用户问起。"
        "如果记忆与当前问题无关，就忽略。"
    )
    temperature: float = 0.7
    max_tokens: int = 1024
    mock: bool = False


def _load_dotenv(path: str | os.PathLike = ".env") -> None:
    """极简 .env 加载（不依赖 python-dotenv），只设未定义的环境变量。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v


def chat(messages: list[dict], config: LLMConfig) -> str:
    """调 OpenAI 兼容 /chat/completions，返回 assistant 文本。

    messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    mock 模式下返回固定回复（含收到的消息数，便于验证闭环）。
    """
    if config.mock:
        n = len([m for m in messages if m["role"] == "user"])
        last = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last = m["content"][:40]
                break
        return (
            f"[mock 回复] 收到 {n} 条 user 消息，最后一条: “{last}…”。"
            f"这是未接 LLM 的占位回复，用于验证记忆闭环。"
        )

    url = config.api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def chat_stream(messages: list[dict], config: LLMConfig) -> Generator[str, None, None]:
    """调 OpenAI 兼容 /chat/completions 并开启 stream，逐块 yield 文本。

    config.mock 为 True 时，模拟时延逐字吐出文本。
    """
    if config.mock:
        n = len([m for m in messages if m["role"] == "user"])
        last = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last = m["content"][:40]
                break
        
        text = (
            f"[mock 流式回复] 收到 {n} 条 user 消息，最后一条: “{last}…”。\n"
            f"这是未接真实大模型的 Mock 回复。\n\n"
            f"<think>\n[Mock 思考链]\n"
            f"1. 分析用户的输入并寻找相关语境...\n"
            f"2. 检查长期记忆缓存并确认检索到的关联卡片...\n"
            f"3. 组装最适宜的本地 Mock 回答信息...\n"
            f"思考完成，开始输出内容。\n</think>\n"
            f"这是流式输出测试内容。我们在这里多打印一些汉字，"
            f"以确保打字机流式效果在前端能够以非常丝滑且自然的方式呈现出来！"
        )
        chunk_size = 6
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
            time.sleep(0.04)
        return

    url = config.api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except Exception:
                        continue
    except Exception as e:
        yield f"\n[Stream Error: {e}]\n"


def build_chat_prompt(
    user_text: str,
    memories: list[dict],
    config: LLMConfig,
    history: list[dict] | None = None,
) -> list[dict]:
    """拼对话 prompt: system(含记忆) + 历史 + 当前用户输入。

    memories: memento.query() 返回的 [{id, text, score, ...}, ...]
    history: 之前的对话 [{"role","content"}, ...]（不含本轮）
    """
    sys = config.system_prompt
    if memories:
        lines = ["【相关记忆】（来自过往对话，按相关性排序）"]
        for i, m in enumerate(memories, 1):
            lines.append(f"{i}. [相关性 {m.get('score', 0):.2f}] {m['text']}")
        sys = sys + "\n\n" + "\n".join(lines)

    messages = [{"role": "system", "content": sys}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


# ─── 配置持久化 ────────────────────────────────────────────

# 基于 llm.py 位置算出项目根，配置文件放 data/ 下，不受启动 cwd 影响
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_FILE = str(_PROJECT_ROOT / "data" / "chat_llm_settings.json")


def load_config_from_file(path: str | os.PathLike = SETTINGS_FILE) -> dict | None:
    """从 JSON 文件读 LLM 配置。不存在返回 None。

    结构: {"api_base", "api_key", "model"}（都可选）。
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_config_to_file(
    cfg: dict,
    path: str | os.PathLike = SETTINGS_FILE,
) -> str:
    """把 LLM 配置写 JSON 文件，返回路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def load_llm_config(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    mock: Optional[bool] = None,
    settings_file: str | os.PathLike = SETTINGS_FILE,
) -> LLMConfig:
    """加载 LLM 配置。

    显式 mock=True 或 CHAT_LLM_MOCK=1 必须强制 mock，避免 demo 启动脚本
    因本地已保存 key 而意外调用真实 LLM。
    """
    _load_dotenv()
    file_cfg = load_config_from_file(settings_file) or {}

    cfg = LLMConfig()

    # api_base / api_key: 显式 > 文件 > CHAT_LLM_* > OPENCODE_API_* > SILICONFLOW_API_*
    cfg.api_base = (
        api_base
        or file_cfg.get("api_base")
        or os.environ.get("CHAT_LLM_API_BASE")
        or os.environ.get("OPENCODE_API_BASE")
        or os.environ.get("SILICONFLOW_API_BASE")
        or ""
    )
    cfg.api_key = (
        api_key
        or file_cfg.get("api_key")
        or os.environ.get("CHAT_LLM_API_KEY")
        or os.environ.get("OPENCODE_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
        or ""
    )
    cfg.model = model or file_cfg.get("model") or os.environ.get("CHAT_LLM_MODEL") or ""

    # mock 判定:
    #   - 显式 --mock 或 CHAT_LLM_MOCK=1 必须强制 mock
    #   - 配置完整时默认使用真 LLM
    #   - 配置不完整时自动回退 mock，保证开源 demo 可直接跑通
    forced_mock = os.environ.get("CHAT_LLM_MOCK", "").lower() in ("1", "true", "yes")
    if mock is True or forced_mock:
        cfg.mock = True
    elif cfg.api_key and cfg.model and cfg.api_base:
        cfg.mock = False
    else:
        cfg.mock = True

    return cfg


def fetch_models(api_base: str, api_key: str) -> list[str]:
    """调 OpenAI 兼容 /models 端点，返回模型 id 列表。

    失败时抛 RuntimeError（含 HTTP 状态或错误信息），由调用方转成友好提示。
    """
    if not api_base:
        raise RuntimeError("未配置 API Base URL")
    url = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # OpenAI 标准格式: {"data": [{"id": "..."}, ...]}
    items = data.get("data") or data.get("models") or []
    if isinstance(items, list):
        ids = [it.get("id") or it.get("model") for it in items if isinstance(it, dict)]
        return sorted([i for i in ids if i])
    # 某些端点直接返回 ["model1", "model2"]
    if isinstance(items, list) and items and isinstance(items[0], str):
        return sorted(items)
    return []
