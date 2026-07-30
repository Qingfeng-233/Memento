"""
Memento 主 API 接口

对外提供记忆系统的读写接口：
  - add_node()             存入记忆（批量模式下暂存）
  - build_index()          批量构建向量索引
  - activate()             激活一组节点（情境共现建边）
  - build_keyword_edges()  基于 keyatten 关键词重叠建边
  - query()                检索 + 扩散联想
  - link()                 主动建边
  - mark_important()       标记重要性
  - trigger_sleep()        触发睡眠周期
  - clock_step()           时钟步推进
"""

from collections import defaultdict

import json
import os
import math
import re
from typing import List, Optional, Dict
from pathlib import Path

import numpy as np

from memento.models import Node
from memento.concept.concept_graph import (
    ConceptGraph,
    cosine_matrix,
    exponential_edge_weight,
)
from memento.index.vector_index import VectorIndex
from memento.graph.memory_graph import MemoryGraph
from memento.engine.diffusion import DiffusionEngine
from memento.engine.decay import DecayEngine
from memento.engine.sleep import SleepEngine, SleepReport


def _extract_context(text: str, keyword: str, window: int = 80) -> str:
    """提取关键词周围的上下文窗口，用于边上下文向量编码。

    取关键词前后各 window 个字符，截断到句子边界（句号/换行）。
    找不到关键词时返回整段文本的前 200 字符。
    """
    pos = text.find(keyword)
    if pos < 0:
        return text[:200]
    start = max(0, pos - window)
    end = min(len(text), pos + len(keyword) + window)
    # 尝试截到句子边界
    for sep in ("。", "！", "？", "\n", ".", "!", "?"):
        s = text.rfind(sep, start, pos)
        if s >= 0 and s > start + 10:
            start = s + 1
        e = text.find(sep, pos + len(keyword), end)
        if e >= 0 and e < end - 10:
            end = e + 1
    ctx = text[start:end].strip()
    return ctx if ctx else text[:200]


def _default_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Memento:
    """
    双系统联想记忆引擎

    记忆 A: 向量快速匹配 (RAG)
    记忆 B: 节点图联想网络 + 激活扩散
    """

    def __init__(
        self,
        embedding_model: str = "tfidf-svd",
        device: str = None,
        # 扩散参数
        diffusion_hops: int = 2,
        diffusion_alpha: float = 0.3,
        diffusion_beta: float = 0.6,
        # 衰减参数
        decay_rate: float = 0.01,
        edge_decay: float = 0.005,
        # 分数缩放
        score_temperature: float = 0.0,
        score_rescale: str = "none",
        # 睡眠巩固 LLM/融合开关（默认关，需显式开启）
        sleep_llm_curate: bool = False,
        sleep_fusion: bool = False,
        llm_curate_cosine: float = 0.35,
        llm_max_calls_per_cycle: int = 50,
        fusion_cosine_threshold: float = 0.85,
    ):
        # 初始化向量索引
        self.vector_index = VectorIndex(
            model_name=embedding_model,
            device=device,
            score_temperature=score_temperature,
            score_rescale=score_rescale,
        )

        # 初始化记忆图
        self.graph = MemoryGraph()

        # 初始化引擎
        self.diffusion = DiffusionEngine(
            self.graph,
            self.vector_index,
            hops=diffusion_hops,
            alpha=diffusion_alpha,
            beta=diffusion_beta,
        )
        self.decay = DecayEngine(
            self.graph, decay_rate=decay_rate, edge_decay=edge_decay
        )
        self.sleep_engine = SleepEngine(
            self.graph,
            self.vector_index,
            self.decay,
            llm_enabled=sleep_llm_curate,
            enable_fusion=sleep_fusion,
            llm_curate_cosine=llm_curate_cosine,
            llm_max_calls_per_cycle=llm_max_calls_per_cycle,
            fusion_cosine_threshold=fusion_cosine_threshold,
        )

        self._clock_step = 0
        # 批量加载缓冲区
        self._pending_nodes: List[dict] = []
        self._index_built = False
        self._next_node_seq = 0

        # 关键词系统（延迟初始化）
        self._keyword_extractor = None
        self._node_keywords: Dict[str, List[str]] = {}
        self._node_keyword_surprisal: Dict[str, Dict[str, float]] = {}
        self._surprisal_model = None  # 惊奇度计算标记（复用主编码器）
        self.concept_graph = ConceptGraph()
        # query_with_concepts 用的预缓存（build_concept_graph 时填充）
        self._concept_ids_cached: list = []
        self._concept_vecs_cached = np.empty((0, 0), dtype=np.float32)

    # ─── 读写接口 ─────────────────────────────────────────────

    def _allocated_node_ids(self) -> set[str]:
        ids = set(self.graph.nodes)
        ids.update(p["id"] for p in self._pending_nodes if p.get("id"))
        ids.update(nid for nid in self.vector_index._id_map if nid)
        return ids

    def _bump_next_node_seq(self, node_id: str) -> None:
        match = re.fullmatch(r"mem_(\d+)", node_id or "")
        if match:
            self._next_node_seq = max(self._next_node_seq, int(match.group(1)) + 1)

    def _generate_node_id(self) -> str:
        allocated = self._allocated_node_ids()
        while True:
            node_id = f"mem_{self._next_node_seq:06d}"
            self._next_node_seq += 1
            if node_id not in allocated:
                return node_id

    def add_node(
        self,
        text: str,
        node_id: str = None,
        importance: float = 0.5,
        tags: list = None,
        source: str = "import",
        created_at: str = None,
    ) -> str:
        """
        存入一个新记忆节点

        在批量模式下（build_index 未调用前），节点暂存到缓冲区。
        调用 build_index() 后，向量索引生效，可以查询。
        """
        if node_id is None:
            node_id = self._generate_node_id()
        else:
            self._bump_next_node_seq(node_id)

        if node_id in self.graph.nodes or any(
            p.get("id") == node_id for p in self._pending_nodes
        ):
            return node_id

        # 暂存到缓冲区
        self._pending_nodes.append(
            {
                "id": node_id,
                "text": text,
                "importance": importance,
                "tags": tags or [],
                "source": source,
                "created_at": created_at,
            }
        )
        return node_id

    def build_index(self):
        """
        批量构建向量索引

        首次调用会构建索引；后续调用只编码新增 pending 节点并追加到
        现有 FAISS 索引，避免全量重算。
        """
        if not self._pending_nodes:
            return

        node_ids = [p["id"] for p in self._pending_nodes]
        texts = [p["text"] for p in self._pending_nodes]

        if self._index_built:
            vectors = self.vector_index.encode(texts, mode="document")
            self.vector_index.add_batch(node_ids, vectors)
        else:
            # 首次批量拟合并构建 FAISS 索引
            self.vector_index.fit_and_add(node_ids, texts)
            vectors = [
                self.vector_index.get_node_vector(node_id) for node_id in node_ids
            ]

        # 回填向量到节点并加入图
        for p, vector in zip(self._pending_nodes, vectors):
            node = Node(
                id=p["id"],
                text=p["text"],
                vector=vector,
                importance=p["importance"],
                vitality=1.0,
                tags=p["tags"],
                source=p["source"],
                created_at=p.get("created_at"),
                status=p.get("status", "active"),
                superseded_by=p.get("superseded_by"),
            )
            self.graph.add_node(node)

        count = len(self._pending_nodes)
        self._pending_nodes.clear()
        self._index_built = True
        return count

    def add_node_live(
        self,
        text: str,
        node_id: str = None,
        importance: float = 0.5,
        tags: list = None,
        source: str = "import",
        created_at: str = None,
    ) -> str:
        """
        实时添加节点（索引已构建后可用）

        使用已拟合的 TF-IDF 模型编码新文本。
        """
        if not self._index_built:
            raise RuntimeError("请先调用 build_index() 构建索引")

        if node_id is None:
            node_id = self._generate_node_id()
        else:
            self._bump_next_node_seq(node_id)
        if node_id in self.graph.nodes or any(
            p.get("id") == node_id for p in self._pending_nodes
        ):
            return node_id

        vector = self.vector_index.encode([text])[0]
        node = Node(
            id=node_id,
            text=text,
            vector=vector,
            importance=importance,
            vitality=1.0,
            tags=tags or [],
            source=source,
            created_at=created_at,
        )
        self.graph.add_node(node)
        self.vector_index.add(node_id, vector)
        return node_id

    def activate(self, node_ids: List[str]):
        """
        激活一组节点 — 情境共现建边

        窗口内所有节点两两之间建立/加强边:
          w_ij += Δw × λ_i × λ_j
        """
        valid_ids = [nid for nid in node_ids if nid in self.graph.nodes]

        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                n_a = self.graph.get_node(valid_ids[i])
                n_b = self.graph.get_node(valid_ids[j])
                delta_w = 0.1 * n_a.vitality * n_b.vitality
                self.graph.add_edge(
                    valid_ids[i], valid_ids[j], weight=delta_w, edge_type="cooccurrence"
                )

        for nid in valid_ids:
            self.decay.boost_vitality(nid, 0.2)

    # ─── 关键词建边 ──────────────────────────────────────────

    def build_keyword_edges(
        self,
        top_k: int = 5,
        min_overlap: int = 1,
        max_node_freq: int = 20,
        weight_per_keyword: float = 0.15,
        max_weight: float = 0.6,
        semantic_filter: bool = False,
        min_cos_sim: float = 0.30,
        compute_surprisal: bool = False,
        surprisal_tolerance: float = None,
        min_surprisal: float = None,
        surprisal_top_k: int = None,
        surprisal_model: str = "thenlper/gte-small-zh",
        keyword_model: str = "models/Qwen3-Embedding-0.6B",
        keyword_device: str = None,
        keyword_dtype: str | None = "float16",
        keyword_cache_enabled: bool = True,
        keyword_cache_dir: str | os.PathLike = "data/keyatten_cache",
    ) -> dict:
        """基于 keyatten 关键词重叠建边

        对所有已有节点提取关键词，通过倒排索引高效找到共享关键词的节点对，
        建立关键词重叠边。

        可选:
          - semantic_filter: 边级过滤，只保留两端向量语义相近的关键词边
          - compute_surprisal: 计算每个关键词的惊奇度 (1 - cos)，用 gte-small-zh 编码
          - min_surprisal: 只保留两端惊奇度都 >= 阈值的共享关键词（高惊奇度=独特锚点）
          - surprisal_tolerance: 惊奇度容差匹配，两端差值超过此值的不计入
          - surprisal_top_k: 节点内按惊奇度排序，只保留 top-K 个关键词参与建边

        Args:
            top_k: 每个节点提取的关键词数
            min_overlap: 最少共享关键词数才建边
            max_node_freq: 关键词出现在超过此数量的节点中则跳过
            weight_per_keyword: 每个共享关键词贡献的权重
            max_weight: 边权重上限
            semantic_filter: 是否启用语义交叉过滤
            min_cos_sim: 语义过滤的余弦相似度阈值
            compute_surprisal: 是否计算关键词惊奇度分数
            surprisal_tolerance: 惊奇度容差，设值后自动开启惊奇度计算
            min_surprisal: 惊奇度下限，只保留两端惊奇度都 >= 此值的关键词。
                低惊奇度=主题内常见词，高惊奇度=独特锚点词。推荐 0.5~0.7
            surprisal_top_k: 节点内惊奇度 top-K，每个节点只保留惊奇度最高的 K 个
                关键词参与建边（相对排名，不受绝对阈值影响）。推荐 2~3
            surprisal_model: 惊奇度编码器模型（mean pooling，适合短文本）
            keyword_model: keyatten 使用的模型路径
            keyword_device: 关键词模型设备（默认跟随 vector_index）
            keyword_dtype: keyatten 模型 dtype，默认 float16
            keyword_cache_enabled: 是否启用 keyatten 内部缓存
            keyword_cache_dir: keyatten 内部缓存目录

        Returns:
            {"edges_added": int, "edges_rejected": int,
             "kw_surprisal_rejected": int, "kw_topk_rejected": int,
             "total_keywords": int, "vocab_size": int,
             "node_keywords": dict,
             "surprisal": dict | None}
        """
        if not self._index_built:
            raise RuntimeError("请先调用 build_index() 构建索引")

        # 延迟加载关键词提取器
        if self._keyword_extractor is None:
            from memento.index.keyatten_extractor import MemoryKeywordExtractor

            device = keyword_device or _default_torch_device()
            self._keyword_extractor = MemoryKeywordExtractor(
                model_path=keyword_model,
                device=device,
                default_top_k=top_k,
                dtype=keyword_dtype,
                cache_enabled=keyword_cache_enabled,
                cache_dir=keyword_cache_dir,
            )
            # 恢复磁盘保存的 IDF
            if hasattr(self, "_saved_idf") and self._saved_idf:
                self._keyword_extractor.set_idf(self._saved_idf)
                del self._saved_idf

        # 收集所有节点 ID 和文本
        all_nodes = sorted(self.graph.nodes.items(), key=lambda x: x[0])
        node_ids = [nid for nid, _ in all_nodes]
        texts = [node.text for _, node in all_nodes]

        # 建立 IDF 基线
        vocab_size = self._keyword_extractor.update_idf(texts)

        # 逐节点提取关键词
        self._node_keywords = {}
        for nid, text in zip(node_ids, texts):
            self._node_keywords[nid] = self._keyword_extractor.extract(
                text, top_k=top_k
            )

        # 惊奇度计算（可选，surprisal_tolerance / min_surprisal / surprisal_top_k 时自动开启）
        if (
            surprisal_tolerance is not None
            or min_surprisal is not None
            or surprisal_top_k is not None
        ):
            compute_surprisal = True

        surprisal_data = None
        if compute_surprisal:
            import numpy as np

            # 直接用主编码器（Qwen3）计算惊奇度，关键词和文本在同一向量空间
            all_kws = list(
                set(kw for kws in self._node_keywords.values() for kw in kws)
            )
            kw_vecs = self.vector_index.encode(all_kws, mode="document")
            kw_to_vec = dict(zip(all_kws, kw_vecs))

            # 节点向量已经有了，直接复用
            nid_to_text_vec = {}
            for nid in node_ids:
                node = self.graph.get_node(nid)
                if node is not None and node.vector is not None:
                    nid_to_text_vec[nid] = node.vector

            self._node_keyword_surprisal = {}
            for nid in node_ids:
                text_vec = nid_to_text_vec.get(nid)
                if text_vec is None:
                    continue
                scores = {}
                for kw in self._node_keywords.get(nid, []):
                    vec = kw_to_vec.get(kw)
                    if vec is not None:
                        cos = float(np.dot(text_vec, vec))
                        scores[kw] = round(1.0 - cos, 4)
                self._node_keyword_surprisal[nid] = scores

            surprisal_data = dict(self._node_keyword_surprisal)

        # 节点内惊奇度 top-K 裁剪：每个节点只保留惊奇度最高的 top_k 个关键词参与建边
        kw_topk_rejected = 0
        if surprisal_top_k is not None and self._node_keyword_surprisal:
            for nid in list(self._node_keywords.keys()):
                scores = self._node_keyword_surprisal.get(nid, {})
                if not scores:
                    continue
                # 按惊奇度降序取 top-K
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_kws = set(kw for kw, _ in ranked[:surprisal_top_k])
                kw_topk_rejected += len(self._node_keywords[nid]) - len(top_kws)
                self._node_keywords[nid] = [
                    kw for kw in self._node_keywords[nid] if kw in top_kws
                ]

        # 倒排索引：keyword → [node_ids]
        kw_to_nodes = defaultdict(list)
        for nid, kws in self._node_keywords.items():
            for kw in kws:
                kw_to_nodes[kw].append(nid)

        # 基于关键词重叠建边
        edge_set = set()
        added, rejected = 0, 0
        kw_surprisal_rejected = 0
        for kw, nids in kw_to_nodes.items():
            if len(nids) < 2 or len(nids) > max_node_freq:
                continue
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    pair = tuple(sorted([nids[i], nids[j]]))
                    if pair in edge_set:
                        continue
                    edge_set.add(pair)

                    # 语义交叉过滤（可选）
                    if semantic_filter:
                        na = self.graph.get_node(pair[0])
                        nb = self.graph.get_node(pair[1])
                        if (
                            na is None
                            or nb is None
                            or na.vector is None
                            or nb.vector is None
                        ):
                            rejected += 1
                            continue
                        import numpy as np

                        cos = float(np.dot(na.vector, nb.vector))
                        if cos < min_cos_sim:
                            rejected += 1
                            continue

                    shared = set(self._node_keywords[pair[0]]) & set(
                        self._node_keywords[pair[1]]
                    )

                    # 惊奇度容差匹配（可选）
                    if surprisal_tolerance is not None:
                        original_count = len(shared)
                        matched = []
                        for kw in shared:
                            s_a = self._node_keyword_surprisal.get(pair[0], {}).get(kw)
                            s_b = self._node_keyword_surprisal.get(pair[1], {}).get(kw)
                            if s_a is not None and s_b is not None:
                                if abs(s_a - s_b) <= surprisal_tolerance:
                                    matched.append(kw)
                        kw_surprisal_rejected += original_count - len(matched)
                        shared = set(matched)

                    # 惊奇度下限过滤（可选）：只保留两端都是高惊奇度的关键词
                    if min_surprisal is not None:
                        original_count = len(shared)
                        high_surprisal = []
                        for kw in shared:
                            s_a = self._node_keyword_surprisal.get(pair[0], {}).get(kw)
                            s_b = self._node_keyword_surprisal.get(pair[1], {}).get(kw)
                            if (
                                s_a is not None
                                and s_b is not None
                                and s_a >= min_surprisal
                                and s_b >= min_surprisal
                            ):
                                high_surprisal.append(kw)
                        kw_surprisal_rejected += original_count - len(high_surprisal)
                        shared = set(high_surprisal)

                    if len(shared) < min_overlap:
                        continue
                    weight = len(shared) * weight_per_keyword
                    self.graph.add_edge(
                        pair[0],
                        pair[1],
                        weight=min(weight, max_weight),
                        edge_type="keyword",
                    )
                    added += 1

        return {
            "edges_added": added,
            "edges_rejected": rejected,
            "kw_surprisal_rejected": kw_surprisal_rejected,
            "kw_topk_rejected": kw_topk_rejected,
            "total_keywords": sum(len(v) for v in self._node_keywords.values()),
            "vocab_size": vocab_size,
            "node_keywords": dict(self._node_keywords),
            "surprisal": surprisal_data,
        }

    def get_node_keywords(self, node_id: str) -> List[str]:
        """获取节点的关键词列表"""
        return self._node_keywords.get(node_id, [])

    def get_keyword_surprisal(self, node_id: str) -> Dict[str, float]:
        """获取节点各关键词的惊奇度分数 (1 - cos_sim)

        分数越高表示关键词对这段文本越"意外"。
        需先在 build_keyword_edges() 中设置 compute_surprisal=True。
        """
        return self._node_keyword_surprisal.get(node_id, {})

    def build_event_edges_from_keywords(
        self,
        min_overlap: int = 1,
        max_node_freq: int = 20,
        weight_per_keyword: float = 0.15,
        max_weight: float = 0.6,
    ) -> dict:
        """复用已提取的统计关键词，为事件节点建立联想边。

        该方法复用 ``build_concept_graph()`` 填充的 ``_node_keywords``，
        不再次加载 KeyAtten 或其他模型。概念图会过滤泛词，而事件图仍可
        使用两条记忆共同出现的有效关键词作为经验连接。
        """
        if not self._node_keywords:
            return {
                "edges_added": 0,
                "candidate_pairs": 0,
                "shared_keywords": 0,
            }

        keyword_to_events: dict[str, list[str]] = defaultdict(list)
        for event_id, keywords in self._node_keywords.items():
            for keyword in set(keywords):
                keyword_to_events[keyword].append(event_id)

        pair_shared: dict[tuple[str, str], set[str]] = defaultdict(set)
        for keyword, event_ids in keyword_to_events.items():
            event_ids = sorted(event_ids)
            if len(event_ids) < 2 or len(event_ids) > max_node_freq:
                continue
            for i, source in enumerate(event_ids):
                for target in event_ids[i + 1 :]:
                    pair_shared[(source, target)].add(keyword)

        added = 0
        for pair, shared_keywords in pair_shared.items():
            if len(shared_keywords) < min_overlap:
                continue
            self.graph.add_edge(
                pair[0],
                pair[1],
                weight=min(len(shared_keywords) * weight_per_keyword, max_weight),
                edge_type="keyword",
            )
            added += 1

        return {
            "edges_added": added,
            "candidate_pairs": len(pair_shared),
            "shared_keywords": sum(len(items) for items in pair_shared.values()),
        }

    # ─── 关键词副节点图 ─────────────────────────────────────

    def build_concept_graph(
        self,
        top_k: int = 8,
        keyword_method: str = "keyatten",
        max_node_freq: int = 30,
        keyword_sim_threshold: float = 0.65,
        keyword_temperature: float = 0.08,
        keyword_top_neighbors: int = 5,
        event_concept_weight_scale: float = 1.0,
        phrase_top_k: int = 3,
        max_concepts: int = 300,
        min_concept_energy: float = 0.5,
        dedup_concepts: bool = False,
        dedup_threshold: float = 0.90,
        # LLM surprisal（惊奇度）作为 event-concept 边权重
        use_surprisal: bool = False,
        surprisal_model: str = "models/Qwen/Qwen3.5-4B",
        surprisal_cache_dir: str | os.PathLike = "data/surprisal_cache",
        keyword_model: str = "models/Qwen3-Embedding-0.6B",
        keyword_device: str = None,
        keyword_dtype: str | None = "float16",
        keyword_cache_enabled: bool = True,
        keyword_cache_dir: str | os.PathLike = "data/keyatten_cache",
    ) -> dict:
        """构建关键词副节点图

        关键词作为独立副节点:
          - 事件 -> 关键词: 根据关键词初始能量建边
          - 关键词 -> 关键词: 根据向量相似度建边，边权指数化

        不做事件上下文染色；关键词向量只使用关键词自身文本。
        min_concept_energy 用于过滤“女生/男生/游戏”这类相对高频泛词。

        dedup_concepts 开启后，在 add_concept 之前对关键词做余弦贪心去重
        (dedup_threshold)，把 "服务配置" / "环境配置缺失" 这类近义锚点合并成
        同一个概念，doc_freq 相加，避免概念图冗余紧簇和 doc_freq 拆碎。
        """
        if not self._index_built:
            raise RuntimeError("请先调用 build_index() 构建索引")

        all_nodes = sorted(self.graph.nodes.items(), key=lambda x: x[0])
        node_ids = [nid for nid, _ in all_nodes]
        texts = [node.text for _, node in all_nodes]
        total_docs = len(texts)

        self._node_keywords = {}
        doc_freq: dict[str, int] = {}
        if keyword_method == "keyatten":
            if self._keyword_extractor is None:
                from memento.index.keyatten_extractor import MemoryKeywordExtractor

                device = keyword_device or _default_torch_device()
                self._keyword_extractor = MemoryKeywordExtractor(
                    model_path=keyword_model,
                    device=device,
                    dtype=keyword_dtype,
                    default_top_k=top_k,
                    cache_enabled=keyword_cache_enabled,
                    cache_dir=keyword_cache_dir,
                )
            vocab_size = self._keyword_extractor.update_idf(texts)
            for nid, text in zip(node_ids, texts):
                keywords = self._keyword_extractor.extract(text, top_k=top_k)
                self._node_keywords[nid] = keywords
                for kw in set(keywords):
                    doc_freq[kw] = doc_freq.get(kw, 0) + 1
        else:
            from memento.index.keyword_extractor import KeywordExtractor

            extractor = KeywordExtractor(method="statistical")
            extractor.fit_corpus(texts)
            vocab_size = len(extractor._corpus_doc_freq)
            phrase_doc_freq = self._build_phrase_doc_freq(texts)
            for nid, text in zip(node_ids, texts):
                keywords = [kw for kw, _ in extractor.extract(text, top_k=top_k)]
                keywords.extend(
                    self._extract_phrase_keywords(
                        text,
                        phrase_doc_freq,
                        total_docs=total_docs,
                        anchor_keywords=keywords,
                        top_k=phrase_top_k,
                    )
                )
                keywords = list(dict.fromkeys(keywords))
                self._node_keywords[nid] = keywords
                for kw in set(keywords):
                    doc_freq[kw] = doc_freq.get(kw, 0) + 1

        # 先按 doc_freq 粗筛（去掉过高频和零频）
        candidate_keywords = sorted(
            kw for kw, freq in doc_freq.items() if 0 < freq <= max_node_freq
        )
        # 编码所有候选词的向量（供后续建图复用）
        if candidate_keywords:
            cand_vectors = self.vector_index.encode(candidate_keywords, mode="document")
        else:
            cand_vectors = []

        # 预计算情感词标记（zipf 惩罚由 initial_energy 内部用 wordfreq 查）
        kw_is_emotion: dict[str, bool] = {
            kw: ConceptGraph.is_emotion_word(kw) for kw in candidate_keywords
        }

        # 用新公式（个人 IDF × 外部 zipf 惩罚 × 情感豁免）过滤
        usable_keywords = sorted(
            kw
            for kw in candidate_keywords
            if ConceptGraph.initial_energy(
                kw,
                doc_freq[kw],
                total_docs,
                is_emotion=kw_is_emotion.get(kw, False),
            )
            >= min_concept_energy
        )
        if max_concepts and len(usable_keywords) > max_concepts:
            usable_keywords = sorted(
                usable_keywords,
                key=lambda kw: ConceptGraph.initial_energy(
                    kw,
                    doc_freq[kw],
                    total_docs,
                    is_emotion=kw_is_emotion.get(kw, False),
                ),
                reverse=True,
            )[:max_concepts]
            usable_keywords = sorted(usable_keywords)
        # 取最终 usable_keywords 对应的向量
        # O(1) dict 查找替代 O(K) list.index()（原写法是 O(K²)）
        cand_index = {kw: i for i, kw in enumerate(candidate_keywords)}
        keyword_vectors = (
            [cand_vectors[cand_index[kw]] for kw in usable_keywords]
            if usable_keywords
            else []
        )
        kw_to_vector = dict(zip(usable_keywords, keyword_vectors))

        # 关键词近义去重（可选）：合并 "服务配置" / "环境配置缺失" 这类高余弦锚点。
        # 在 add_concept 之前重写 usable_keywords / doc_freq / per-node 关键词，
        # 使下游概念创建、事件链接、概念边都看到规范形式，一次性修掉三个失效模式。
        dedup_merge_log = []
        if dedup_concepts and len(usable_keywords) >= 2:
            stacked = np.vstack(keyword_vectors)
            remap = ConceptGraph.dedup_concept_keywords(
                usable_keywords, stacked, threshold=dedup_threshold
            )
            if any(kw != canon for kw, canon in remap.items()):
                dedup_merge_log = sorted(
                    [
                        {"from": kw, "to": canon}
                        for kw, canon in remap.items()
                        if kw != canon
                    ],
                    key=lambda m: (m["to"], m["from"]),
                )
                # doc_freq 合并到规范词
                new_doc_freq: dict[str, int] = {}
                for kw, freq in doc_freq.items():
                    canon = remap.get(kw, kw)
                    new_doc_freq[canon] = new_doc_freq.get(canon, 0) + freq
                doc_freq = new_doc_freq
                # 规范词集（向量取规范词自身，因为更具体的词往往也是更好锚点）
                usable_keywords = sorted(set(remap.values()))
                # per-node 关键词重映射 + 去重（保持原顺序）
                for nid, kws in list(self._node_keywords.items()):
                    seen = set()
                    remapped = []
                    for kw in kws:
                        canon = remap.get(kw, kw)
                        if canon not in seen:
                            seen.add(canon)
                            remapped.append(canon)
                    self._node_keywords[nid] = remapped

        concept_graph = ConceptGraph()
        for kw in usable_keywords:
            concept_graph.add_concept(
                kw,
                kw_to_vector.get(kw),
                doc_freq[kw],
                total_docs,
                precomputed_energy=ConceptGraph.initial_energy(
                    kw,
                    doc_freq[kw],
                    total_docs,
                    is_emotion=kw_is_emotion.get(kw, False),
                ),
            )

        event_concept_edges = 0
        # LLM surprisal：每条 event-concept 边的权重由关键词在这篇记忆里的
        # 意外度调节。surprisal 高（意外）→ 边强；surprisal 低（预期内）→ 边弱。
        surprisal_calc = None
        if use_surprisal:
            from memento.engine.surprisal_calculator import SurprisalCalculator

            surprisal_calc = SurprisalCalculator(
                model_path=surprisal_model,
                cache_dir=surprisal_cache_dir,
            )

        # 预收集所有 (event, keyword) 的上下文文本，批量编码
        # 上下文 = 关键词周围的文本窗口，编码后作为边向量，
        # 查询时与 query 向量做余弦相似度实现"查询感知路由"。
        ctx_pairs: list[tuple[str, str, str]] = []  # (nid, kw, context_text)
        for nid, keywords in self._node_keywords.items():
            node = self.graph.get_node(nid)
            text = node.text if node else ""
            for kw in keywords:
                if kw not in kw_to_vector:
                    continue
                ctx = _extract_context(text, kw)
                ctx_pairs.append((nid, kw, ctx))

        # 批量编码上下文（一次 GPU forward pass）
        ctx_vectors: dict[tuple[str, str], np.ndarray] = {}
        if ctx_pairs:
            ctx_texts = [c for _, _, c in ctx_pairs]
            encoded = self.vector_index.encode(ctx_texts, mode="document")
            for (nid, kw, _), vec in zip(ctx_pairs, encoded):
                ctx_vectors[(nid, concept_graph.concept_id(kw))] = vec

        for nid, keywords in self._node_keywords.items():
            # 一次性算这篇记忆里所有关键词的 surprisal（缓存命中则秒回）
            surprisal_map = {}
            if surprisal_calc is not None:
                node = self.graph.get_node(nid)
                if node and node.text:
                    usable_kws = [kw for kw in keywords if kw in kw_to_vector]
                    if usable_kws:
                        surprisal_map = surprisal_calc.compute(node.text, usable_kws)

            for kw in keywords:
                if kw not in kw_to_vector:
                    continue
                cid = concept_graph.concept_id(kw)
                concept = concept_graph.concepts[cid]
                # 基础权重 = 节点 energy
                weight = concept.initial_energy * event_concept_weight_scale
                # surprisal 调节：surprisal 高（意外）→ 满权；
                # surprisal 低（预期内/泛词）→ 边权大幅降低。
                # 关键：用乘法 factor，让 surprisal 低时能真正压低边权，
                # 而不是加法偏移后 clamp 到 1.0 抹平差异。
                if surprisal_calc is not None and kw in surprisal_map:
                    raw_s = surprisal_map[kw]["first"]
                    # factor = min(1.0, surprisal / 5.0)
                    # surprisal=0 → 0.0（完全不连，词太预期内）
                    # surprisal=2 → 0.4（弱连）
                    # surprisal=5 → 1.0（满权）
                    # surprisal=15 → 1.0（还是满权，专有名词）
                    s_factor = min(1.0, raw_s / 5.0)
                    weight = weight * s_factor
                concept_graph.link_event_concept(
                    nid,
                    cid,
                    min(1.0, weight),
                    context_vector=ctx_vectors.get((nid, cid)),
                )
                event_concept_edges += 1

        concept_edges = 0
        if usable_keywords:
            vectors = np.vstack([kw_to_vector[kw] for kw in usable_keywords])
            sims = cosine_matrix(vectors)
            for i, kw in enumerate(usable_keywords):
                cid = concept_graph.concept_id(kw)
                candidates = []
                for j, other_kw in enumerate(usable_keywords):
                    if i == j:
                        continue
                    sim = float(sims[i, j])
                    weight = exponential_edge_weight(
                        sim,
                        threshold=keyword_sim_threshold,
                        temperature=keyword_temperature,
                    )
                    if weight > 0:
                        candidates.append((other_kw, weight))
                candidates.sort(key=lambda item: item[1], reverse=True)
                for other_kw, weight in candidates[:keyword_top_neighbors]:
                    other_id = concept_graph.concept_id(other_kw)
                    # 只在当前节点没有这条边时计数（link_concepts 取 max）
                    is_new = other_id not in concept_graph.concept_edges.get(cid, {})
                    concept_graph.link_concepts(cid, other_id, weight)
                    if is_new:
                        concept_edges += 1

        self.concept_graph = concept_graph
        # 预排序扩散邻居（diffuse 每个 hop 直接查表，不再 sorted()）
        concept_graph.presort_neighbors(top_neighbors=keyword_top_neighbors)
        # 预缓存概念向量矩阵 + ID 列表，query_with_concepts 直接用，
        # 避免每次查询 np.vstack + list comprehension。
        if concept_graph.concepts:
            items = list(concept_graph.concepts.items())
            self._concept_ids_cached = [cid for cid, _ in items]
            self._concept_vecs_cached = np.vstack([c.vector for _, c in items]).astype(
                np.float32
            )
            # 预归一化（查询时不需要再算 norm）
            norms = np.linalg.norm(self._concept_vecs_cached, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._concept_vecs_cached = self._concept_vecs_cached / norms
        else:
            self._concept_ids_cached = []
            self._concept_vecs_cached = np.empty((0, 0), dtype=np.float32)
        return {
            "concepts": len(concept_graph.concepts),
            "event_concept_edges": event_concept_edges,
            "concept_edges": concept_edges,
            "vocab_size": vocab_size,
            "keyword_method": keyword_method,
            "max_concepts": max_concepts,
            "min_concept_energy": min_concept_energy,
            "keyword_sim_threshold": keyword_sim_threshold,
            "keyword_temperature": keyword_temperature,
            "dedup_concepts": dedup_concepts,
            "dedup_threshold": dedup_threshold if dedup_concepts else None,
            "dedup_merges": dedup_merge_log,
            "dedup_merge_count": len(dedup_merge_log),
        }

    @staticmethod
    def _iter_chinese_phrases(text: str) -> set[str]:
        """生成短中文锚点候选，用于保留“Docker Compose”这类短语。"""
        phrases: set[str] = set()
        for span in re.findall(r"[\u4e00-\u9fff]{3,32}", text[:1500]):
            max_len = min(8, len(span))
            for size in range(3, max_len + 1):
                for start in range(0, len(span) - size + 1):
                    phrase = span[start : start + size]
                    if Memento._is_bad_phrase(phrase):
                        continue
                    phrases.add(phrase)
        return phrases

    @staticmethod
    def _is_bad_phrase(phrase: str) -> bool:
        bad_prefixes = (
            "为什么",
            "这个",
            "那个",
            "就是",
            "其实",
            "如果",
            "因为",
            "所以",
            "而你",
            "但是",
            "然后",
            "这里",
            "那里",
            "一种",
            "这个",
        )
        bad_suffixes = ("什么", "如何", "怎么", "时候", "一点", "一样")
        if phrase.startswith(bad_prefixes) or phrase.endswith(bad_suffixes):
            return True
        if len(set(phrase)) <= 1:
            return True
        return False

    @classmethod
    def _build_phrase_doc_freq(cls, texts: list[str]) -> dict[str, int]:
        doc_freq: dict[str, int] = {}
        for text in texts:
            for phrase in cls._iter_chinese_phrases(text):
                doc_freq[phrase] = doc_freq.get(phrase, 0) + 1
        return doc_freq

    @classmethod
    def _extract_phrase_keywords(
        cls,
        text: str,
        phrase_doc_freq: dict[str, int],
        total_docs: int,
        anchor_keywords: list[str],
        top_k: int = 3,
    ) -> list[str]:
        candidates = cls._iter_chinese_phrases(text)
        anchors = [kw for kw in anchor_keywords if len(kw) >= 2]
        forced = cls._anchor_span_phrases(text, anchors)
        scored = []
        for phrase in candidates:
            df = phrase_doc_freq.get(phrase, 0)
            if df <= 0:
                continue
            anchor_hits = sum(1 for kw in anchors if kw in phrase and kw != phrase)
            if anchor_hits == 0:
                continue
            idf = math.log((total_docs + 1) / (df + 1)) + 1.0
            length_bonus = min(1.0, len(phrase) / 6)
            anchor_bonus = 1.0 + 0.5 * anchor_hits
            scored.append((phrase, idf * length_bonus * anchor_bonus))
        scored.sort(key=lambda item: item[1], reverse=True)
        result = list(dict.fromkeys(forced))
        for phrase, _ in scored:
            if phrase not in result:
                result.append(phrase)
            if len(result) >= top_k:
                break
        return result

    @staticmethod
    def _anchor_span_phrases(text: str, anchors: list[str]) -> list[str]:
        """保留能同时覆盖多个关键词的最短短语。"""
        results = []
        unique_anchors = list(dict.fromkeys(anchors))
        for i, left in enumerate(unique_anchors):
            for right in unique_anchors[i + 1 :]:
                left_pos = text.find(left)
                right_pos = text.find(right)
                if left_pos < 0 or right_pos < 0:
                    continue
                start = min(left_pos, right_pos)
                end = max(left_pos + len(left), right_pos + len(right))
                phrase = text[start:end]
                if 3 <= len(phrase) <= 8 and re.fullmatch(r"[\u4e00-\u9fff]+", phrase):
                    if not Memento._is_bad_phrase(phrase):
                        results.append(phrase)
        return results[:3]

    def query_with_concepts(
        self,
        text: str,
        k: int = 10,
        seed_k: int = 20,
        concept_k: int = 8,
        concept_hops: int = 2,
        concept_weight: float = 0.35,
        debug: bool = False,
        debug_top_concepts: int = 10,
    ) -> List[Dict]:
        """向量 RAG + 关键词副节点扩散检索"""
        if not self.concept_graph.concepts:
            raise RuntimeError("请先调用 build_concept_graph()")

        query_vector = self.vector_index.encode([text], mode="query")[0]
        rag_hits = self.vector_index.search(query_vector, k=seed_k)
        rag_scores = {node_id: max(0.0, score) for node_id, score in rag_hits}

        # 使用 build_concept_graph 时预缓存的概念向量矩阵
        concept_ids = self._concept_ids_cached
        concept_vecs = self._concept_vecs_cached
        q = query_vector.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        sims = (concept_vecs @ q.T).reshape(-1)
        seed_concepts = {}
        seed_debug = []
        for idx in np.argsort(-sims)[:concept_k]:
            cid = concept_ids[int(idx)]
            concept = self.concept_graph.concepts[cid]
            sim = float(sims[int(idx)])
            activation = max(0.0, sim) * concept.initial_energy
            seed_concepts[cid] = activation
            if debug:
                seed_debug.append(
                    {
                        "concept_id": cid,
                        "concept": concept.text,
                        "similarity": round(sim, 4),
                        "initial_energy": round(float(concept.initial_energy), 4),
                        "activation": round(float(activation), 4),
                        "doc_freq": concept.doc_freq,
                    }
                )

        concept_activations = self.concept_graph.diffuse(
            seed_concepts,
            hops=concept_hops,
        )
        concept_event_scores = self.concept_graph.events_from_concepts(
            concept_activations,
            query_vector=query_vector,  # 查询感知路由
        )
        event_supports = (
            self.concept_graph.event_supports_from_concepts(concept_activations)
            if debug
            else {}
        )
        if concept_event_scores:
            max_concept_score = max(concept_event_scores.values())
            if max_concept_score > 0:
                concept_event_scores = {
                    node_id: score / max_concept_score
                    for node_id, score in concept_event_scores.items()
                }

        all_ids = set(rag_scores) | set(concept_event_scores)
        merged = []
        for node_id in all_ids:
            rag_score = rag_scores.get(node_id, 0.0)
            concept_score = concept_event_scores.get(node_id, 0.0)
            # 乘法辅助：概念分只能给已有的 RAG 相关性锦上添花，
            # 不能把 RAG 低分的笔记拉到高位。
            # rag=0 的纯概念结果自然被过滤（final=0）。
            final = rag_score * (1.0 + concept_weight * concept_score)
            node = self.graph.get_node(node_id)
            # 状态过滤：跳过 superseded（融合源）/ dormant / cold 节点，
            # 避免已被融合或遗忘的记忆从检索漏回。
            if node and node.status == "active":
                merged.append((node_id, final, rag_score, concept_score))
        merged.sort(key=lambda item: item[1], reverse=True)

        output = []
        for node_id, final, rag_score, concept_score in merged[:k]:
            node = self.graph.get_node(node_id)
            output.append(
                {
                    "id": node_id,
                    "text": node.text,
                    "score": round(final, 4),
                    "rag_score": round(rag_score, 4),
                    "concept_score": round(concept_score, 4),
                    "importance": round(node.importance, 4),
                    "vitality": round(node.vitality, 4),
                    "edges": node.edge_count,
                    "tags": node.tags,
                }
            )
            if debug:
                output[-1]["concept_supports"] = event_supports.get(node_id, [])[
                    :debug_top_concepts
                ]
        if debug:
            activated_debug = []
            for cid, activation in sorted(
                concept_activations.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:debug_top_concepts]:
                concept = self.concept_graph.concepts.get(cid)
                if concept is None:
                    continue
                activated_debug.append(
                    {
                        "concept_id": cid,
                        "concept": concept.text,
                        "activation": round(float(activation), 4),
                        "doc_freq": concept.doc_freq,
                        "initial_energy": round(float(concept.initial_energy), 4),
                    }
                )
            return {
                "query": text,
                "seed_concepts": seed_debug,
                "activated_concepts": activated_debug,
                "results": output,
            }
        return output

    def query_associative(
        self,
        text: str,
        k: int = 10,
        seed_k: int = 5,
        event_weight: float = 0.6,
        concept_weight: float = 0.4,
    ) -> List[Dict]:
        """融合事件图扩散与概念图扩散的完整在线联想查询。

        ``query()`` 负责从少量 RAG 种子沿事件边扩散；
        ``query_with_concepts()`` 负责概念副节点激活与查询感知路由。
        两路分数分别归一化后融合，任一路都可以把候选带入结果集。
        """
        if not self._index_built:
            raise RuntimeError("请先调用 build_index()")
        if not self.concept_graph.concepts:
            event_hits = self.query(text, k=k, seed_k=seed_k)
            rag_by_id = {
                hit["id"]: hit["score"]
                for hit in self.query_rag_only(text, k=max(k, seed_k))
            }
            max_event = max((hit["score"] for hit in event_hits), default=0.0)
            return [
                {
                    **hit,
                    "score": round(
                        hit["score"] / max_event if max_event > 0 else 0.0,
                        4,
                    ),
                    "rag_score": round(rag_by_id.get(hit["id"], 0.0), 4),
                    "event_score": round(
                        hit["score"] / max_event if max_event > 0 else 0.0,
                        4,
                    ),
                    "concept_score": 0.0,
                }
                for hit in event_hits
            ]

        candidate_k = min(self.graph.node_count, max(k * 4, 20))
        seed_k = min(self.graph.node_count, max(1, seed_k))
        event_hits = self.query(text, k=candidate_k, seed_k=seed_k)
        concept_hits = self.query_with_concepts(
            text,
            k=candidate_k,
            seed_k=candidate_k,
        )

        max_event = max((hit["score"] for hit in event_hits), default=0.0)
        max_concept = max((hit["score"] for hit in concept_hits), default=0.0)
        event_by_id = {hit["id"]: hit for hit in event_hits}
        concept_by_id = {hit["id"]: hit for hit in concept_hits}

        merged = []
        for node_id in set(event_by_id) | set(concept_by_id):
            event_hit = event_by_id.get(node_id, {})
            concept_hit = concept_by_id.get(node_id, {})
            event_score = (
                event_hit.get("score", 0.0) / max_event if max_event > 0 else 0.0
            )
            concept_route_score = (
                concept_hit.get("score", 0.0) / max_concept
                if max_concept > 0
                else 0.0
            )
            final_score = (
                event_weight * event_score + concept_weight * concept_route_score
            )
            node = self.graph.get_node(node_id)
            if node is None or node.status != "active" or final_score <= 0:
                continue
            merged.append(
                {
                    "id": node_id,
                    "text": node.text,
                    "score": round(final_score, 4),
                    "rag_score": round(concept_hit.get("rag_score", 0.0), 4),
                    "event_score": round(event_score, 4),
                    "concept_score": round(
                        concept_hit.get("concept_score", 0.0), 4
                    ),
                    "importance": round(node.importance, 4),
                    "vitality": round(node.vitality, 4),
                    "edges": node.edge_count,
                    "tags": node.tags,
                }
            )

        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:k]

    def link_concepts(self, source: str, target: str, weight: float = 0.8) -> None:
        """手工连接两个关键词副节点。不存在时自动创建无向量副节点。"""
        for text in (source, target):
            cid = self.concept_graph.concept_id(text)
            if cid not in self.concept_graph.concepts:
                vector = None
                if self._index_built:
                    vector = self.vector_index.encode([text], mode="document")[0]
                self.concept_graph.add_concept(
                    text=text,
                    vector=vector,
                    doc_freq=1,
                    total_docs=max(1, self.graph.node_count),
                )
        self.concept_graph.link_concepts(
            self.concept_graph.concept_id(source),
            self.concept_graph.concept_id(target),
            weight,
        )

    def query(self, text: str, k: int = 10, seed_k: int = 20) -> List[Dict]:
        """
        查询：RAG 检索 + 扩散联想
        """
        query_vector = self.vector_index.encode([text], mode="query")[0]
        results = self.diffusion.query(query_vector, k=k, seed_k=seed_k)

        output = []
        for node_id, score in results:
            node = self.graph.get_node(node_id)
            # 双保险：diffusion 内部已按 active 过滤，这里再确认一次
            if node and node.status == "active":
                output.append(
                    {
                        "id": node_id,
                        "text": node.text,
                        "score": round(score, 4),
                        "importance": round(node.importance, 4),
                        "vitality": round(node.vitality, 4),
                        "edges": node.edge_count,
                        "tags": node.tags,
                    }
                )
        return output

    def query_rag_only(self, text: str, k: int = 10) -> List[Dict]:
        """纯 RAG 查询（不扩散），用于对比"""
        query_vector = self.vector_index.encode([text], mode="query")[0]
        seeds = self.vector_index.search(query_vector, k=k)

        output = []
        for node_id, sim in seeds:
            node = self.graph.get_node(node_id)
            # 状态过滤：superseded / dormant / cold 不进检索
            if node and node.status == "active":
                output.append(
                    {
                        "id": node_id,
                        "text": node.text,
                        "score": round(sim, 4),
                        "importance": round(node.importance, 4),
                        "vitality": round(node.vitality, 4),
                        "edges": node.edge_count,
                        "tags": node.tags,
                    }
                )
        return output

    def link(self, node_a: str, node_b: str, weight: float = 0.8):
        """主动关联建边"""
        self.graph.add_edge(node_a, node_b, weight=weight, edge_type="manual")

    def mark_important(self, node_id: str, importance: float = 1.0):
        """标记节点为重要"""
        node = self.graph.get_node(node_id)
        if node:
            node.importance = min(1.0, max(0.0, importance))

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.graph.get_node(node_id)

    def clock_step(self):
        """推进一个时钟步"""
        self.decay.step()
        self._clock_step += 1

    def trigger_sleep(self) -> SleepReport:
        """触发睡眠周期"""
        report = self.sleep_engine.run_sleep_cycle()
        self._clock_step += 1
        return report

    # ─── 统计与持久化 ────────────────────────────────────────

    @property
    def stats(self) -> dict:
        active = sum(1 for n in self.graph.nodes.values() if n.status == "active")
        dormant = sum(1 for n in self.graph.nodes.values() if n.status == "dormant")
        cold = sum(1 for n in self.graph.nodes.values() if n.status == "cold")
        return {
            "total_nodes": self.graph.node_count,
            "active_nodes": active,
            "dormant_nodes": dormant,
            "cold_nodes": cold,
            "total_edges": self.graph.edge_count,
            "vector_index_size": self.vector_index.size,
            "clock_step": self._clock_step,
            "keyword_nodes": len(self._node_keywords),
        }

    def save(self, directory: str):
        """保存记忆系统到磁盘"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        nodes_data = {}
        for nid, node in self.graph.nodes.items():
            nodes_data[nid] = {
                "id": node.id,
                "text": node.text,
                "importance": node.importance,
                "vitality": node.vitality,
                "access_count": node.access_count,
                "edge_count": node.edge_count,
                "tags": node.tags,
                "source": node.source,
                "status": node.status,
                "created_at": node.created_at,
                "superseded_by": node.superseded_by,
                "fused_from": node.fused_from,
            }
        with open(path / "nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)

        edges_data = []
        for src, tgt, edge in self.graph.get_all_edges():
            edges_data.append(
                {
                    "source": src,
                    "target": tgt,
                    "weight": edge.weight,
                    "type": edge.edge_type,
                }
            )
        with open(path / "edges.json", "w", encoding="utf-8") as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)

        import faiss

        if self.vector_index.index is not None:
            faiss.write_index(self.vector_index.index, str(path / "vectors.faiss"))
        with open(path / "id_map.json", "w", encoding="utf-8") as f:
            json.dump(self.vector_index._id_map, f)

        # 持久化 TF-IDF+SVD 拟合管道，让 load 后能 encode 新查询/增量节点
        # （其他后端 qwen3/st/api 从磁盘或 API 重新加载，无需此处处理）
        if (
            self.vector_index._backend == "tfidf-svd"
            and self.vector_index._pipeline is not None
        ):
            import joblib

            joblib.dump(self.vector_index._pipeline, str(path / "tfidf_pipeline.pkl"))

        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "clock_step": self._clock_step,
                    "index_built": self._index_built,
                    "pending_nodes": self._pending_nodes,
                    "next_node_seq": self._next_node_seq,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 保存关键词数据
        if self._node_keywords:
            kw_data = {
                "node_keywords": self._node_keywords,
                "idf": (
                    self._keyword_extractor.get_idf()
                    if self._keyword_extractor
                    else None
                ),
                "surprisal": self._node_keyword_surprisal or None,
            }
            with open(path / "keywords.json", "w", encoding="utf-8") as f:
                json.dump(kw_data, f, ensure_ascii=False, indent=2)

        # 保存概念图（含查询感知路由用的边上下文向量）。
        # 让 load() 后可直接 query_with_concepts，无需重跑 keyatten。
        if self.concept_graph.concepts:
            self.concept_graph.save(path)

    def load(self, directory: str):
        """从磁盘加载记忆系统"""
        import numpy as np
        import faiss as faiss_lib

        path = Path(directory)

        with open(path / "nodes.json", "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
        for nid, d in nodes_data.items():
            node = Node(
                id=d["id"],
                text=d["text"],
                importance=d["importance"],
                vitality=d["vitality"],
                access_count=d.get("access_count", 0),
                edge_count=d.get("edge_count", 0),
                tags=d.get("tags", []),
                source=d.get("source", "import"),
                status=d.get("status", "active"),
                created_at=d.get("created_at"),
                superseded_by=d.get("superseded_by"),
                fused_from=d.get("fused_from", []),
            )
            self.graph.add_node(node)

        # vectors.faiss 可选：索引未构建时（仅有 pending 节点）不存在
        if (path / "vectors.faiss").exists():
            self.vector_index.index = faiss_lib.read_index(str(path / "vectors.faiss"))
            with open(path / "id_map.json", "r", encoding="utf-8") as f:
                self.vector_index._id_map = json.load(f)
            self.vector_index._is_fitted = True

            for i, nid in enumerate(self.vector_index._id_map):
                node = self.graph.get_node(nid)
                if node:
                    node.vector = self.vector_index.index.reconstruct(i)

            # 恢复 TF-IDF+SVD 管道（load 后 encode 新查询/增量节点需要）
            if (path / "tfidf_pipeline.pkl").exists():
                import joblib

                self.vector_index._pipeline = joblib.load(
                    str(path / "tfidf_pipeline.pkl")
                )
                # 同步真实向量维度（小数据集下 SVD 维度 < 构造默认 128）
                self.vector_index._dimension = self.vector_index.index.d
        else:
            self.vector_index._is_fitted = False

        with open(path / "edges.json", "r", encoding="utf-8") as f:
            edges_data = json.load(f)
        for ed in edges_data:
            self.graph.add_edge(
                ed["source"],
                ed["target"],
                weight=ed["weight"],
                edge_type=ed.get("type", "cooccurrence"),
            )

        # 恢复 pending 节点 + 索引状态（让 CLI add→save→add→build 循环可恢复）
        index_built_flag = (path / "vectors.faiss").exists()
        if (path / "meta.json").exists():
            with open(path / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._clock_step = meta.get("clock_step", 0)
            self._pending_nodes = meta.get("pending_nodes", []) or []
            self._next_node_seq = meta.get("next_node_seq", 0)
            index_built_flag = meta.get("index_built", index_built_flag)

        # 恢复关键词数据
        if (path / "keywords.json").exists():
            with open(path / "keywords.json", "r", encoding="utf-8") as f:
                kw_data = json.load(f)
            self._node_keywords = kw_data.get("node_keywords", {})
            self._node_keyword_surprisal = kw_data.get("surprisal", {}) or {}
            # IDF 延迟恢复：需要关键词提取器初始化后才能 set
            self._saved_idf = kw_data.get("idf")

        # 恢复概念图（含边上下文向量）。旧 store 无此文件时返回 False，
        # 沿用原行为（query_with_concepts 需手动 build_concept_graph）。
        if self.concept_graph.load(path):
            # 重建 query_with_concepts 用的预缓存概念向量矩阵
            if self.concept_graph.concepts:
                items = list(self.concept_graph.concepts.items())
                self._concept_ids_cached = [cid for cid, _ in items]
                self._concept_vecs_cached = (
                    np.vstack(
                        [c.vector for _, c in items if c.vector is not None]
                    ).astype(np.float32)
                    if any(c.vector is not None for _, c in items)
                    else np.empty((0, 0), dtype=np.float32)
                )
                norms = np.linalg.norm(self._concept_vecs_cached, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._concept_vecs_cached = (
                    self._concept_vecs_cached / norms
                    if self._concept_vecs_cached.size
                    else self._concept_vecs_cached
                )

        self._index_built = index_built_flag
        for nid in self._allocated_node_ids():
            self._bump_next_node_seq(nid)
