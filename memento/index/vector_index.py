"""
向量索引层 - 记忆 A：快速语义匹配 (RAG)

支持四种嵌入方案：
  1. tfidf-svd (默认): TF-IDF + SVD 降维 + FAISS，无需下载模型
  2. sentence-transformers: 加载 ST 预训练模型 (如 MiniLM-L12)
  3. qwen3: Qwen3-Embedding 系列 (0.6B / 4B / 8B)
     - last_token_pool (官方推荐) + L2 normalize
     - 支持 query/document instruction prefix
  4. api: 远程 API embedding（如 SiliconFlow）
     - model_name 格式: "api:Qwen/Qwen3-Embedding-4B"
     - 自动从环境变量读取 SILICONFLOW_API_BASE / SILICONFLOW_API_KEY

嵌入模型作为数学函数（信号转换），不参与推理。
"""

import hashlib
import json
import os
from pathlib import Path
import re
from typing import List, Tuple, Optional

import faiss
import numpy as np


class VectorIndex:
    """向量索引：支持 TF-IDF+SVD、Sentence-Transformer、Qwen3 三种后端"""

    def __init__(
        self,
        model_name: str = "tfidf-svd",
        device: str = None,
        dimension: int = 128,
        score_temperature: float = 0.0,
        score_rescale: str = "none",
    ):
        self._model_name = model_name
        self._device = device
        self._dimension = dimension
        self._score_temperature = score_temperature  # 参数（温度/幂次/对比度）
        self._score_rescale = score_rescale  # none / softmax / power / stretch
        self._pipeline = None  # TF-IDF+SVD pipeline
        self._model = None  # ST / Qwen3 model
        self._tokenizer = None  # Qwen3 tokenizer
        self._torch_device = None  # torch device
        self._index = None  # FAISS index
        self._id_map: List[str] = []
        self._is_fitted = False
        self._api_cache_dir = None

        if model_name == "tfidf-svd":
            self._backend = "tfidf-svd"
        elif model_name.startswith("api:"):
            self._backend = "api"
            self._api_model = model_name[4:]  # e.g. "Qwen/Qwen3-Embedding-4B"
        elif "qwen" in model_name.lower():
            self._backend = "qwen3"
        else:
            self._backend = "sentence-transformers"

    # ─── TF-IDF ─────────────────────────────────────────────

    def _build_tfidf_pipeline(self, n_components: int = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        if n_components is None:
            n_components = min(self._dimension, 2000)
        self._pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, sublinear_tf=True)),
                ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ]
        )

    def _tokenize(self, text: str) -> str:
        import jieba

        words = jieba.lcut(text)
        words = [w.strip() for w in words if len(w.strip()) > 1]
        return " ".join(words) if words else text

    # ─── Qwen3-Embedding ────────────────────────────────────

    # 官方 instruction prefix 格式（Qwen3-Embedding 系列专用）
    # 4B/8B 训练时用了前缀，0.6B 没用——通过 _use_prefix 自动切换
    QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

    def _load_qwen3(self):
        """加载 Qwen3-Embedding 模型，自动检测 GPU 和模型规模"""
        import torch
        from transformers import AutoTokenizer, AutoModel

        # 自动检测：优先用用户指定的 device，否则自动选 GPU/CPU
        if self._device is not None:
            self._torch_device = torch.device(self._device)
        elif torch.cuda.is_available():
            self._torch_device = torch.device("cuda")
            print(f"  检测到 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._torch_device = torch.device("cpu")

        # 判断模型规模：4B/8B 用 instruction prefix，0.6B 不用
        model_lower = self._model_name.lower()
        self._use_prefix = (
            "4b" in model_lower
            or "8b" in model_lower
            or "4B" in self._model_name
            or "8B" in self._model_name
        )
        model_tag = (
            "4B"
            if "4b" in model_lower or "4B" in self._model_name
            else "8B"
            if "8b" in model_lower or "8B" in self._model_name
            else "0.6B"
        )

        print(
            f"  加载 Qwen3-Embedding-{model_tag} -> {self._torch_device}"
            f"{' (启用 instruction prefix)' if self._use_prefix else ''}"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True, padding_side="left"
        )
        self._model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self._torch_device)
        self._model.eval()
        print(f"  模型加载完成, device={self._torch_device}")

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """Last token pooling — Qwen3-Embedding 官方推荐方法

        取每个序列最后一个有效 token 的隐藏状态作为整句嵌入。
        对于 left-padded 的 batch，直接取 [:, -1]。
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _encode_qwen3(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 8192,
        mode: str = "document",
    ) -> np.ndarray:
        """Qwen3-Embedding 编码: last_token_pool + L2 normalize

        Args:
            texts: 要编码的文本列表
            batch_size: 批次大小 (4B 建议 4~8)
            max_length: 最大 token 长度 (4B 支持到 32K)
            mode: "document" 或 "query"（4B/8B 模型 query 会加 instruction prefix）
        """
        import torch

        # 4B/8B 模型给 query 加 instruction prefix
        if mode == "query" and self._use_prefix:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]

        all_vecs = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._torch_device)

            with torch.no_grad():
                outputs = self._model(**batch)

            # last_token_pool -> [B, H]
            pooled = self._last_token_pool(
                outputs.last_hidden_state.float(), batch["attention_mask"]
            )
            # L2 normalize
            norm = pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
            normalized = pooled / norm
            all_vecs.append(normalized.cpu().numpy())

        return np.vstack(all_vecs).astype(np.float32)

    def encode_query_qwen3(self, query: str, max_length: int = 256) -> np.ndarray:
        """编码查询文本（加 instruction prefix）"""
        return self._encode_qwen3([query], mode="query", max_length=max_length)[0]

    # ─── API Embedding (SiliconFlow etc.) ────────────────────

    def _load_api(self):
        """初始化 API embedding 后端"""
        self._api_base = os.environ.get(
            "SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1"
        )
        self._api_key = os.environ.get("SILICONFLOW_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("API embedding 需要设置 SILICONFLOW_API_KEY 环境变量")
        root = Path(os.environ.get("MEMENTO_CACHE_DIR", "data/vector_cache"))
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._api_model)
        self._api_cache_dir = root / model_slug
        self._api_cache_dir.mkdir(parents=True, exist_ok=True)
        # 用 instruction prefix（同 Qwen3-Embedding 4B/8B 规范）
        model_lower = self._api_model.lower()
        self._use_prefix = "4b" in model_lower or "8b" in model_lower
        print(f"  API Embedding: {self._api_model} @ {self._api_base}")
        print(f"  API Embedding cache: {self._api_cache_dir}")

    def _api_cache_key(self, text: str, mode: str) -> str:
        payload = {
            "model": self._api_model,
            "base": self._api_base,
            "mode": mode,
            "use_prefix": bool(self._use_prefix),
            "text": text,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _api_cache_path(self, text: str, mode: str) -> Path:
        if self._api_cache_dir is None:
            self._load_api()
        return self._api_cache_dir / f"{self._api_cache_key(text, mode)}.npy"

    def _encode_api_uncached(
        self, texts: List[str], batch_size: int = 100, mode: str = "document"
    ) -> np.ndarray:
        """通过远程 API 批量编码文本（不读写缓存）

        Args:
            texts: 文本列表
            batch_size: 每批请求的文本数（API 限制通常 100~200）
            mode: "document" 或 "query"（4B/8B 模型 query 会加 instruction prefix）
        """
        import time
        import requests

        # query 模式加 instruction prefix
        if mode == "query" and self._use_prefix:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]

        all_vecs = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            last_error = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        f"{self._api_base}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self._api_model, "input": batch_texts},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt == 2:
                        raise
                    sleep_seconds = 2**attempt
                    print(f"  API embedding retry {attempt + 1}/3 after error: {exc}")
                    time.sleep(sleep_seconds)
            data = resp.json()
            # 按 index 排序确保顺序正确
            items = sorted(data["data"], key=lambda x: x["index"])
            for item in items:
                all_vecs.append(item["embedding"])

        vecs = np.array(all_vecs, dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def _encode_api(
        self, texts: List[str], batch_size: int = 100, mode: str = "document"
    ) -> np.ndarray:
        """通过远程 API 编码文本，按文本 hash 做磁盘缓存"""
        if self._api_cache_dir is None:
            self._load_api()

        vectors = [None] * len(texts)
        misses = []
        miss_indices = []

        for idx, text in enumerate(texts):
            cache_path = self._api_cache_path(text, mode)
            if cache_path.exists():
                try:
                    vectors[idx] = np.load(cache_path).astype(np.float32)
                    continue
                except Exception:
                    cache_path.unlink(missing_ok=True)
            misses.append(text)
            miss_indices.append(idx)

        if misses:
            print(f"  API cache miss: {len(misses)}/{len(texts)} ({mode})")
            miss_vecs = self._encode_api_uncached(
                misses, batch_size=batch_size, mode=mode
            )
            for original_idx, text, vec in zip(miss_indices, misses, miss_vecs):
                vectors[original_idx] = vec.astype(np.float32)
                np.save(self._api_cache_path(text, mode), vectors[original_idx])
        else:
            print(f"  API cache hit: {len(texts)}/{len(texts)} ({mode})")

        return np.vstack(vectors).astype(np.float32)

    def encode_token_level(
        self, text: str, max_length: int = 512
    ) -> "tuple[np.ndarray, np.ndarray, list]":
        """获取 token 级别的嵌入向量，用于惊奇关键词提取

        Returns:
            token_embeddings: [seq_len, hidden_size] L2 归一化后的 token 向量
            mean_embedding: [hidden_size] 句子的平均向量
            tokens: token 文本列表（已清理特殊符号）
        """
        import torch

        encoded = self._tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self._torch_device)

        with torch.no_grad():
            outputs = self._model(**encoded, output_hidden_states=True)

        hidden = outputs.last_hidden_state[0].float()  # [seq_len, hidden_size]
        mask = encoded["attention_mask"][0].float()  # [seq_len]

        # mean pooling
        mean_emb = (hidden * mask.unsqueeze(-1)).sum(dim=0) / mask.sum().clamp(min=1e-9)
        mean_norm = mean_emb / mean_emb.norm().clamp(min=1e-12)

        # L2 normalize each token
        tok_norms = hidden.norm(dim=1, keepdim=True).clamp(min=1e-12)
        tok_normalized = hidden / tok_norms

        # decode token texts
        input_ids = encoded["input_ids"][0]
        token_texts = []
        for tid in input_ids:
            t = self._tokenizer.decode([tid])
            # 清理 BPE 空格标记
            t = t.replace("Ġ", "").replace("▁", "").replace(" ", "")
            token_texts.append(t)

        # 过滤 padding 和特殊 token
        valid_mask = mask.bool().cpu().numpy()
        tok_embeddings = tok_normalized.cpu().numpy()[valid_mask]
        token_texts = [token_texts[i] for i, v in enumerate(valid_mask) if v]
        # 跳过首尾的特殊 token (BOS/EOS)
        if len(token_texts) > 2:
            tok_embeddings = tok_embeddings[1:-1]
            token_texts = token_texts[1:-1]

        return tok_embeddings, mean_norm.cpu().numpy(), token_texts

    # ─── 统一接口 ────────────────────────────────────────────

    def fit_and_add(self, node_ids: List[str], texts: List[str]):
        """批量拟合并添加所有向量到 FAISS"""
        if self._backend == "tfidf-svd":
            self._fit_tfidf(node_ids, texts)
        elif self._backend == "api":
            self._fit_api(node_ids, texts)
        elif self._backend == "qwen3":
            self._fit_qwen3(node_ids, texts)
        else:
            self._fit_st(node_ids, texts)

    def _fit_tfidf(self, node_ids: List[str], texts: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        tokenized = [self._tokenize(t) for t in texts]
        # 先单独拟合 TfidfVectorizer 探测词表大小，避免 SVD n_components
        # 超过 n_features（小数据集会触发 sklearn ValueError）。
        probe = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        probe.fit(tokenized)
        n_features = len(probe.vocabulary_)
        # TruncatedSVD 要求 n_components <= min(n_features, n_samples)
        n_components = max(1, min(self._dimension, n_features, len(tokenized)))
        self._build_tfidf_pipeline(n_components=n_components)
        vectors = self._pipeline.fit_transform(tokenized).astype(np.float32)
        self._is_fitted = True
        self._dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._id_map = list(node_ids)

    def _fit_qwen3(self, node_ids: List[str], texts: List[str]):
        self._load_qwen3()
        print(f"  编码 {len(texts)} 条文本 ...")
        vectors = self._encode_qwen3(texts)
        self._is_fitted = True
        self._dimension = vectors.shape[1]
        print(f"  编码完成, 维度={self._dimension}")
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)
        self._id_map = list(node_ids)

    def _fit_st(self, node_ids: List[str], texts: List[str]):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"  加载模型: {self._model_name} ...")
            self._model = SentenceTransformer(self._model_name)
            print(f"  模型加载完成")
        print(f"  编码 {len(texts)} 条文本 ...")
        vectors = self._model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        self._is_fitted = True
        self._dimension = vectors.shape[1]
        print(f"  编码完成, 维度={self._dimension}")
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)
        self._id_map = list(node_ids)

    def _fit_api(self, node_ids: List[str], texts: List[str]):
        """通过远程 API 批量编码并构建 FAISS 索引"""
        self._load_api()
        print(f"  API 编码 {len(texts)} 条文本 ...")
        import time

        t0 = time.time()
        vectors = self._encode_api(texts)
        t_enc = time.time() - t0
        self._is_fitted = True
        self._dimension = vectors.shape[1]
        print(f"  API 编码完成, 维度={self._dimension}, 耗时={t_enc:.1f}s")
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)
        self._id_map = list(node_ids)

    def encode(self, texts: List[str], mode: str = "document") -> np.ndarray:
        """将文本编码为向量（已归一化）

        Args:
            texts: 文本列表
            mode: "document" (建索引用) 或 "query" (查询用，Qwen3 会加 instruction prefix)
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit_and_add() 构建索引")
        if self._backend == "tfidf-svd":
            tokenized = [self._tokenize(t) for t in texts]
            vectors = self._pipeline.transform(tokenized).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vectors / norms
        elif self._backend == "api":
            return self._encode_api(texts, mode=mode)
        elif self._backend == "qwen3":
            return self._encode_qwen3(texts, mode=mode)
        else:
            return self._model.encode(texts, normalize_embeddings=True).astype(
                np.float32
            )

    def add(self, node_id: str, vector: np.ndarray):
        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._id_map.append(node_id)

    def add_batch(self, node_ids: List[str], vectors: np.ndarray):
        vecs = vectors.astype(np.float32).copy()
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._id_map.extend(node_ids)

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        if not self._id_map or self._index is None:
            return []
        k = min(k, len(self._id_map))
        vec = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scores, indices = self._index.search(vec, k)

        # 收集有效结果
        raw_scores = []
        valid_ids = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._id_map):
                raw_scores.append(float(score))
                valid_ids.append(self._id_map[idx])

        # 分数后处理：放大差异，给扩散算法提供梯度
        if self._score_rescale != "none" and raw_scores:
            raw_scores = self._rescale_scores(raw_scores)

        return list(zip(valid_ids, raw_scores))

    def _rescale_scores(self, scores: list) -> list:
        """分数后处理：softmax / power / stretch"""
        arr = np.array(scores, dtype=np.float64)
        method = self._score_rescale
        param = self._score_temperature

        if method == "softmax" and param > 0:
            # softmax 温度缩放
            scaled = arr / param
            scaled -= scaled.max()
            exp_s = np.exp(scaled)
            total = exp_s.sum()
            if total > 0:
                return (exp_s / total).tolist()

        elif method == "power" and param > 1:
            # 幂次放大：score^p，保留排序，拉大差距
            # 先归一化到 [0,1]
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                normed = (arr - mn) / (mx - mn)
            else:
                normed = np.ones_like(arr)
            powered = normed**param
            # 再映射回原始范围
            result = powered * (mx - mn) + mn
            return result.tolist()

        elif method == "stretch":
            # min-max 拉伸 + 对比度增强
            # param 控制对比度 (>1 增强, <1 减弱)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                normed = (arr - mn) / (mx - mn)  # [0, 1]
                if param > 0:
                    # S 型对比度: 1/(1 + exp(-k*(x-0.5)))
                    k = param
                    centered = normed - 0.5
                    sigmoid = 1.0 / (1.0 + np.exp(-k * centered))
                    # 归一化到 [0,1]
                    s_min = 1.0 / (1.0 + np.exp(k * 0.5))
                    s_max = 1.0 / (1.0 + np.exp(-k * 0.5))
                    normed = (sigmoid - s_min) / (s_max - s_min)
                result = normed * (mx - mn) + mn
                return result.tolist()

        return scores

    def get_vector(self, internal_idx: int) -> np.ndarray:
        return self._index.reconstruct(internal_idx)

    def get_node_vector(self, node_id: str) -> np.ndarray:
        if node_id in self._id_map:
            idx = self._id_map.index(node_id)
            return self._index.reconstruct(idx)
        return None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self._id_map)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
