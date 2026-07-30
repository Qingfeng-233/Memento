"""
惊奇关键词提取器 — 模拟人脑的"惊奇感"记忆锚点机制

人脑不需要对所有文字做数学运算。当遇到一大段信息时，几个"刺点"
（不常见的、印象深刻的词）会自然浮现，成为记忆的出发点和检索入口。

本模块实现两种提取方式：
  1. statistical: TF-IDF × 稀有度 × 具体性 （零成本，基于统计）
  2. qwen3-token: 利用 Qwen3 token 级嵌入计算上下文意外度
     — 某个 token 的向量离句子均值越远，"惊奇感"越强

作者：Memento 项目
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import math
import re


class KeywordExtractor:
    """惊奇关键词提取器

    用法:
        extractor = KeywordExtractor(method="statistical")
        extractor.fit_corpus(all_texts)  # 统计语料词频
        keywords = extractor.extract("一大段文字...", top_k=5)
        # => [("容器配置", 0.92), ("虚拟现实", 0.78), ...]
    """

    def __init__(self, method: str = "statistical",
                 vector_index=None):
        """
        Args:
            method: "statistical" | "qwen3-token"
            vector_index: Qwen3 VectorIndex 实例（method="qwen3-token" 时需要）
        """
        self.method = method
        self.vector_index = vector_index
        self._corpus_doc_freq: Dict[str, int] = {}
        self._corpus_doc_count = 0

    def fit_corpus(self, texts: List[str]):
        """统计语料级词频（IDF 计算用）"""
        self._corpus_doc_count = len(texts)
        self._corpus_doc_freq = {}
        for text in texts:
            words = self._segment(text)
            seen = set()
            for w in words:
                if w not in seen:
                    self._corpus_doc_freq[w] = self._corpus_doc_freq.get(w, 0) + 1
                    seen.add(w)
        print(f"  关键词提取器: 语料={self._corpus_doc_count}篇, "
              f"词汇={len(self._corpus_doc_freq)}个")

    def extract(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """从文本中提取惊奇关键词

        Returns:
            [(词, 惊奇分数 0~1), ...]  按分数降序
        """
        if self.method == "qwen3-token" and self.vector_index is not None:
            return self._extract_qwen3_token(text, top_k)
        return self._extract_statistical(text, top_k)

    # ─── 统计方法 ──────────────────────────────────────────

    def _extract_statistical(self, text: str, top_k: int = 5
                              ) -> List[Tuple[str, float]]:
        """TF-IDF × 稀有度 × 具体性 = 惊奇分数

        三个维度模拟人脑筛选机制：
        - TF：这个词在这段文字里有多突出
        - IDF：这个词在整个语料里有多罕见（罕见=记忆锚点价值高）
        - 具体性：长词通常更具体（如"虚拟现实" > "现实"）
        """
        words = self._segment(text)
        if not words:
            return []

        total = len(words)
        word_tf = Counter(words)

        scores = {}
        for word, tf in word_tf.items():
            # (1) TF 占比
            tf_score = tf / total

            # (2) IDF — 越罕见越有价值
            df = self._corpus_doc_freq.get(word, 0)
            idf_score = math.log(
                (self._corpus_doc_count + 1) / (df + 1)
            ) if self._corpus_doc_count > 0 else 1.0

            # (3) 具体性 — 2-4字词最常见，5字以上可能是专有名词/术语
            length = len(word)
            if length <= 1:
                specificity = 0.3
            elif length <= 3:
                specificity = 0.7
            elif length <= 5:
                specificity = 0.85
            else:
                specificity = 1.0

            # (4) 实体信号 — 包含英文/数字的词可能更有信息量
            entity_bonus = 1.0
            if re.search(r'[A-Za-z0-9]', word):
                entity_bonus = 1.3

            saliency = tf_score * idf_score * specificity * entity_bonus
            scores[word] = saliency

        # 排序并归一化
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_items:
            return []
        max_score = sorted_items[0][1]
        if max_score == 0:
            return [(w, 1.0 / (i + 1)) for i, (w, _) in enumerate(sorted_items[:top_k])]
        return [(w, round(s / max_score, 4)) for w, s in sorted_items[:top_k]]

    # ─── Qwen3 Token 级惊奇度 ─────────────────────────────

    def _extract_qwen3_token(self, text: str, top_k: int = 5
                              ) -> List[Tuple[str, float]]:
        """利用 Qwen3 的 token 级嵌入计算上下文意外度

        原理：
        1. 将文本送入 Qwen3，获取每个 token 的向量表示
        2. 计算整句的均值向量（"语义重心"）
        3. 每个 token 与均值的距离 = "上下文意外度"
        4. 距离越远的 token → 在上下文中越"刺眼" → 惊奇关键词

        这模拟了人脑的一个特性：一个词在上下文中越"不合群"，
        越容易被单独记住。
        """
        try:
            tok_embs, mean_emb, token_texts = (
                self.vector_index.encode_token_level(text)
            )
        except Exception:
            # 降级为统计方法
            return self._extract_statistical(text, top_k)

        if len(token_texts) == 0:
            return self._extract_statistical(text, top_k)

        # 计算每个 token 与语义重心的余弦距离
        token_surprises = {}
        for i, (emb, tok) in enumerate(zip(tok_embs, token_texts)):
            # 跳过纯标点和单字符
            if len(tok) < 2:
                continue
            if re.match(r'^[，,。.！!？?、；;：:（）()\[\]【】""''\s\-—…]+$', tok):
                continue
            sim = float(np.dot(emb, mean_emb))
            surprise = 1.0 - sim  # 0=完全一致, 1=完全正交, 2=完全相反
            # 用最大值策略（同一词的不同 token 取最"刺眼"的）
            if tok not in token_surprises or surprise > token_surprises[tok]:
                token_surprises[tok] = surprise

        if not token_surprises:
            return self._extract_statistical(text, top_k)

        # 排序并归一化
        sorted_items = sorted(token_surprises.items(),
                              key=lambda x: x[1], reverse=True)
        max_score = sorted_items[0][1]
        if max_score == 0:
            return [(w, 1.0 / (i + 1)) for i, (w, _) in enumerate(sorted_items[:top_k])]
        return [(w, round(s / max_score, 4)) for w, s in sorted_items[:top_k]]

    # ─── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _segment(text: str) -> List[str]:
        """分词并过滤"""
        import jieba
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip()) >= 2]

    def extract_for_query(self, query: str, top_k: int = 5
                           ) -> List[Tuple[str, float]]:
        """从查询中提取惊奇关键词（用于查询扩展）"""
        return self.extract(query, top_k)

    def extract_for_nodes(self, texts: List[str], top_k: int = 5
                           ) -> List[List[Tuple[str, float]]]:
        """批量提取节点的惊奇关键词"""
        return [self.extract(text, top_k) for text in texts]

    def build_keyword_graph(self, node_keywords: List[List[Tuple[str, float]]],
                             node_ids: List[str],
                             min_overlap: int = 1
                              ) -> List[Tuple[str, str, float, List[str]]]:
        """基于惊奇关键词重叠建边

        两个记忆共享越多的惊奇关键词，它们的边权重越大。

        Returns:
            [(src_id, tgt_id, weight, shared_keywords), ...]
        """
        edges = []
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                kws_i = {w for w, _ in node_keywords[i]}
                kws_j = {w for w, _ in node_keywords[j]}
                shared = kws_i & kws_j
                if len(shared) >= min_overlap:
                    # 权重 = 共享词数 / min(|A|, |B|)
                    weight = len(shared) / min(len(kws_i), len(kws_j))
                    edges.append((node_ids[i], node_ids[j],
                                  round(weight, 4), list(shared)))
        return edges
