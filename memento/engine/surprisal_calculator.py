"""
LLM 惊奇度（surprisal）计算器 —— 基于本地 Qwen3.5-4B 的 token 级 logprob。

核心思路（信息论 surprisal）：
    surprisal(token | context) = -log P(token | context)

工程实现：
    1. 把整篇记忆文本输入 LLM，做一次前向传播，拿到所有位置的 logits
    2. 用 offset_mapping 把关键词（字符串）对齐到 token 序列
    3. 关键词的 surprisal = 它覆盖的所有 token 的 -logprob 之和
    4. 取第一次出现为主（novelty），最大值为辅

一次前向传播即可得到全文所有 token 的 surprisal，不管抽多少个关键词。
结果按 hash(文本+关键词) 磁盘缓存，重复构建不重算。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


class SurprisalCalculator:
    """基于 Qwen3.5-4B 的惊奇度计算器。

    用法：
        calc = SurprisalCalculator(model_path="models/Qwen/Qwen3.5-4B")
        # 一次调用算一篇文本里所有关键词的 surprisal
        results = calc.compute("一段记忆文本", ["钢琴", "USB", "file"])
        # => {"钢琴": {"first": 3.2, "max": 5.1, "count": 3}, ...}
    """

    def __init__(
        self,
        model_path: str = "models/Qwen/Qwen3.5-4B",
        device: str | None = None,
        dtype: str = "float16",
        cache_dir: str | os.PathLike = "data/surprisal_cache",
        cache_enabled: bool = True,
        max_length: int = 2048,
    ) -> None:
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device = None
        self._dtype = dtype
        if device:
            self._device_cfg = device
        else:
            import torch
            self._device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ─── 延迟加载 ───────────────────────────────────────────

    def _load(self):
        """加载模型和 tokenizer（首次调用时）。"""
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                     "float32": torch.float32}
        torch_dtype = dtype_map.get(self._dtype, torch.float16)

        print(f"  加载 SurprisalCalculator: {self.model_path} -> {self._device_cfg}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        # 开启 fast tokenizer 以拿 offset_mapping
        if not self._tokenizer.is_fast:
            print("  ⚠ tokenizer 不是 fast 模式，offset_mapping 可能不可用")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True,
            dtype=torch_dtype, device_map=self._device_cfg,
        )
        self._model.eval()
        self._device = self._model.device
        print(f"  SurprisalCalculator 就绪")

    # ─── 缓存 ───────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        # 缓存粒度：整篇文本一次（所有关键词一起算）
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._cache_key(text)}.json"

    def _cache_get(self, text: str) -> dict | None:
        if not self.cache_enabled:
            return None
        path = self._cache_path(text)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _cache_put(self, text: str, data: dict) -> None:
        if not self.cache_enabled:
            return
        try:
            self._cache_path(text).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ─── 核心：一次前向传播 ─────────────────────────────────

    def _compute_all_logprobs(self, text: str) -> np.ndarray:
        """对整篇文本做一次前向传播，返回每个 token 的 -logprob（surprisal）。

        Returns:
            surprisals: [seq_len] 每个位置的 surprisal。
            token_texts: list[str] 每个 token 的文本。
            offsets: list[(start, end)] 每个 token 的字符偏移。
        （返回三元组）
        """
        import torch

        self._load()

        # tokenize，带 offset_mapping
        encoded = self._tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            truncation=True, max_length=self.max_length,
            add_special_tokens=False,  # 不加 BOS/EOS，纯文本 surprisal
        )
        input_ids = encoded["input_ids"].to(self._device)
        offsets = encoded["offset_mapping"][0].tolist()  # [(start,end), ...]

        with torch.no_grad():
            outputs = self._model(input_ids)

        logits = outputs.logits[0]  # [seq_len, vocab_size]
        # logits[t] 预测的是位置 t+1 的 token
        # 我们要的是位置 t 的 token 的 surprisal = -log P(token_t | tokens[0:t])
        # 即用 logits[t-1] 预测 input_ids[t]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)  # [seq_len-1, vocab]
        target_ids = input_ids[0, 1:]  # 从位置 1 开始
        token_logprobs = log_probs[torch.arange(target_ids.shape[0]), target_ids]
        # surprisal of each token (from position 1 onward)
        surprisals = (-token_logprobs).cpu().numpy()  # [seq_len-1]
        # 位置 0 没有 surprisal（没有上文），补 0
        surprisals = np.concatenate([[0.0], surprisals])

        token_texts = [self._tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        return surprisals, token_texts, offsets

    # ─── 关键词对齐 ─────────────────────────────────────────

    @staticmethod
    def _find_keyword_offsets(text: str, keyword: str) -> list[tuple[int, int]]:
        """找关键词在文本中所有出现的字符区间 [start, end)。"""
        offsets = []
        start = 0
        while True:
            idx = text.find(keyword, start)
            if idx < 0:
                break
            offsets.append((idx, idx + len(keyword)))
            start = idx + 1  # 允许重叠匹配
        return offsets

    @staticmethod
    def _align_to_tokens(
        kw_start: int, kw_end: int,
        token_offsets: list[tuple[int, int]],
    ) -> list[int]:
        """把关键词的字符区间对齐到 token 索引。

        只要 token 的字符区间跟关键词有交集，就纳入。
        """
        indices = []
        for i, (ts, te) in enumerate(token_offsets):
            # token 区间 [ts, te) 跟关键词区间 [kw_start, kw_end) 有交集
            if ts < kw_end and te > kw_start:
                indices.append(i)
        return indices

    # ─── 公共接口 ───────────────────────────────────────────

    def compute(self, text: str, keywords: list[str]) -> dict[str, dict]:
        """计算一篇文本里所有关键词的 surprisal。

        一次前向传播 + 缓存。

        Returns:
            {keyword: {"first": float, "max": float, "count": int, "all": [float]}, ...}
            查不到的关键词不在返回值里。
        """
        # 缓存命中
        cached = self._cache_get(text)
        if cached is not None:
            # 从缓存里提取需要的关键词
            result = {}
            for kw in keywords:
                if kw in cached:
                    result[kw] = cached[kw]
            if len(result) == len(keywords):
                return result
            # 部分命中，仍需计算（但前向传播结果可能也要重算）
            # 简单起见：如果缓存存在但不完整，重新计算全部
            # （缓存是整篇文本级别的，不会出现这种情况，除非关键词列表变了）

        # 前向传播
        surprisals, token_texts, token_offsets = self._compute_all_logprobs(text)

        # 对齐每个关键词
        result = {}
        for kw in keywords:
            kw_positions = self._find_keyword_offsets(text, kw)
            if not kw_positions:
                continue

            all_surps = []
            for kw_start, kw_end in kw_positions:
                token_indices = self._align_to_tokens(kw_start, kw_end, token_offsets)
                if not token_indices:
                    continue
                # 这个位置的关键词 surprisal = 覆盖 token 的 surprisal 之和
                kw_surp = float(sum(surprisals[i] for i in token_indices))
                all_surps.append(kw_surp)

            if not all_surps:
                continue

            result[kw] = {
                "first": all_surps[0],   # 第一次出现（novelty）
                "max": max(all_surps),    # 最大值（辅助）
                "count": len(all_surps),
                "all": [round(s, 4) for s in all_surps],
            }

        # 缓存整篇结果
        self._cache_put(text, result)

        return result

    def compute_single(self, text: str, keyword: str) -> dict | None:
        """计算单个关键词的 surprisal（便捷方法）。"""
        result = self.compute(text, [keyword])
        return result.get(keyword)
