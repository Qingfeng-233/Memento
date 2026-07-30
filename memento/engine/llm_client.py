"""
LLM 客户端 — 记忆层离线巩固专用（严格限定用途）

本模块是记忆层唯一的 LLM 入口，修订了 plan.txt 的“本层零大型 LLM 调用”
约束。严格限定条款（见 docs/ decision 文档）：
  - 仅 SleepEngine 可调用（离线巩固，不进 add/query 热路径）
  - 仅对余弦预筛后的候选对调用（不全文喂 LLM）
  - 按 (text_a, text_b) hash 磁盘缓存，同对永不重判
  - LLM 不可用时优雅降级（返回 none 裁决，跳过该对，不崩）

OpenAI-compatible，复用 embedding 已有的环境变量约定：
  - OPENCODE_API_BASE / OPENCODE_API_KEY  （OpenCode deepseek，对齐评测 key）
  - 回退到 SILICONFLOW_API_BASE / SILICONFLOW_API_KEY
  - MEMENTO_LLM_MODEL 默认 deepseek-v4-flash（对齐评测用 LLM）
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass
class LLMVerdict:
    """LLM 对一对候选的裁决。

    verdict:
      "link"  —— 两条记忆语义相关，应建边
      "merge" —— 两条记忆几乎同一件事，应融合（fused_text 给出合成文本）
      "none"  —— 不相关或重复度不够，什么都不做
      None    —— LLM 不可用 / 跳过（调用方应保守地不做改动）
    """
    verdict: Optional[str]
    reason: str = ""
    weight: float = 0.0        # link 时建议的边权
    fused_text: str = ""       # merge 时合成的文本


class LLMClient:
    """OpenAI-compatible LLM 客户端，带磁盘缓存。"""

    DEFAULT_MODEL = "deepseek-v4-flash"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: str | os.PathLike = "data/llm_cache",
        cache_enabled: bool = True,
        timeout: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> None:
        self.model = model or os.environ.get("MEMENTO_LLM_MODEL", self.DEFAULT_MODEL)
        # base_url / api_key：优先 OpenCode，回退 SiliconFlow
        self.base_url = base_url or os.environ.get(
            "OPENCODE_API_BASE",
            os.environ.get("SILICONFLOW_API_BASE", ""),
        )
        self.api_key = api_key or os.environ.get(
            "OPENCODE_API_KEY",
            os.environ.get("SILICONFLOW_API_KEY", ""),
        )
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.cache_hits = 0
        self.errors = 0

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    # ─── 缓存 ───────────────────────────────────────────────

    def _cache_key(self, text_a: str, text_b: str, task: str) -> str:
        # 规范化：去掉空白/大小写差异，按字典序排序保证对称性
        a = re.sub(r"\s+", " ", text_a).strip().lower()
        b = re.sub(r"\s+", " ", text_b).strip().lower()
        if a > b:
            a, b = b, a
        payload = {"model": self.model, "task": task, "a": a, "b": b}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[dict]:
        if not self.cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _cache_put(self, key: str, data: dict) -> None:
        if not self.cache_enabled:
            return
        try:
            self._cache_path(key).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # ─── 核心调用 ───────────────────────────────────────────

    def _chat(self, system: str, user: str) -> Optional[str]:
        """发起一次 chat completion，返回 assistant 文本；失败返回 None。

        注意：deepseek-v4-flash 是 reasoning 模型，会先在 reasoning_content
        里消耗 token（思维链），再产出最终 content。max_tokens 必须留足
        reasoning + content 两段空间（默认 1200），否则 content 为空。
        """
        if not self.available:
            return None
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self.calls += 1
            data = resp.json()
            choice = data["choices"][0]
            # 被截断（max_tokens 不够覆盖 reasoning+content）→ 视为失败，不缓存
            if choice.get("finish_reason") == "length":
                self.errors += 1
                print("  [LLMClient] 响应被 length 截断，调大 max_tokens")
                return None
            content = choice["message"].get("content")
            return content if content else None
        except Exception as exc:
            self.errors += 1
            print(f"  [LLMClient] 调用失败: {exc}")
            return None

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """从 LLM 输出里抠 JSON（容忍 markdown code fence）。"""
        if not text:
            return None
        # 去 ```json ... ``` 围栏
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = m.group(1) if m else text
        # 兜底：抓第一个 { ... }
        if not candidate.strip().startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if 0 <= start < end:
                candidate = candidate[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    # ─── 任务接口 ───────────────────────────────────────────

    def judge_relation(self, text_a: str, text_b: str,
                       cosine: float = 0.0) -> LLMVerdict:
        """裁决两条记忆是 link / merge / none。

        候选对在调用前已做余弦预筛（由 sleep 层负责），这里 LLM 只看文本。
        """
        key = self._cache_key(text_a, text_b, task="judge_relation")
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return LLMVerdict(
                verdict=cached.get("verdict"),
                reason=cached.get("reason", ""),
                weight=float(cached.get("weight", 0.0)),
                fused_text=cached.get("fused_text", ""),
            )

        system = (
            "你是记忆库的离线整理员。给你两条记忆，判断它们的关系，只回 JSON。"
            "字段：verdict（'link'/'merge'/'none'）、reason（<=30字）、"
            "weight（0~1，link 时建议边权）、fused_text（merge 时合成一条简洁文本）。"
            "link=语义相关应建边；merge=几乎同一件事应合并；none=都不。"
        )
        user = (
            f"记忆A：\n{text_a}\n\n"
            f"记忆B：\n{text_b}\n\n"
            f"（向量余弦 {cosine:.3f}）\n"
            "只回 JSON。"
        )
        raw = self._chat(system, user)
        if raw is None:
            return LLMVerdict(verdict=None, reason="LLM unavailable")

        parsed = self._parse_json(raw)
        if parsed is None:
            return LLMVerdict(verdict=None, reason="parse failed")

        verdict = LLMVerdict(
            verdict=parsed.get("verdict"),
            reason=parsed.get("reason", ""),
            weight=float(parsed.get("weight", 0.0) or 0.0),
            fused_text=parsed.get("fused_text", ""),
        )
        # 规范化 verdict
        if verdict.verdict not in {"link", "merge", "none"}:
            verdict.verdict = "none"
        self._cache_put(key, {
            "verdict": verdict.verdict,
            "reason": verdict.reason,
            "weight": verdict.weight,
            "fused_text": verdict.fused_text,
        })
        return verdict

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
        }
