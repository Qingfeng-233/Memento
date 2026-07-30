"""关键词副节点图。

ConceptGraph 只负责关键词节点、事件-关键词边、关键词-关键词边。
事件本身仍由 MemoryGraph 管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class ConceptNode:
    """关键词副节点。"""

    id: str
    text: str
    vector: np.ndarray | None = None
    initial_energy: float = 0.5
    doc_freq: int = 0
    specificity: float = 0.5
    events: dict[str, float] = field(default_factory=dict)


class ConceptGraph:
    """关键词副节点图。"""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.event_to_concepts: Dict[str, Dict[str, float]] = {}
        self.concept_to_events: Dict[str, Dict[str, float]] = {}
        self.concept_edges: Dict[str, Dict[str, float]] = {}
        # 边上下文向量：{(event_id, concept_id): np.ndarray}
        # 用于查询感知路由：查询向量与边向量的相似度决定能量流向。
        self.context_vectors: Dict[tuple[str, str], np.ndarray] = {}
        # 预排序邻居缓存
        self._sorted_neighbors: Dict[str, list] = {}

    @staticmethod
    def concept_id(text: str) -> str:
        return "kw:" + text.strip().lower()

    # 情感/状态词词表（豁免通用度惩罚）。
    # 这些词在通用语料里高频，但标记个人状态/情感色彩，应保持高 energy。
    # 涵盖：基本情绪、心理状态、人际态度、价值判断。
    EMOTION_WORDS: frozenset = frozenset({
        # 基本情绪
        "开心", "快乐", "高兴", "幸福", "兴奋", "激动", "愉快", "欣喜", "喜悦", "满足",
        "难过", "伤心", "悲伤", "痛苦", "郁闷", "沮丧", "失落", "绝望", "崩溃", "心痛",
        "愤怒", "生气", "恼怒", "烦躁", "厌烦", "讨厌", "憎恨", "怨恨", "委屈",
        "害怕", "恐惧", "焦虑", "担心", "紧张", "不安", "慌张", "胆怯",
        "惊讶", "震惊", "迷茫", "困惑", "犹豫", "纠结", "矛盾",
        # 心理状态
        "孤独", "寂寞", "空虚", "落寞", "怀念", "思念", "想念", "依恋", "眷恋",
        "释然", "坦然", "安心", "平静", "淡然", "麻木", "疲惫", "厌倦", "懈怠",
        "希望", "渴望", "期待", "向往", "憧憬", "梦想", "幻想",
        "愧疚", "后悔", "遗憾", "自责", "羞愧", "尴尬",
        "骄傲", "自豪", "自信", "自卑", "挫败", "无力", "无奈",
        # 人际态度
        "喜欢", "爱", "爱意", "心动", "迷恋", "暗恋", "表白", "依赖",
        "信任", "怀疑", "背叛", "原谅", "嫉炉", "嫉妒",
        "孤独感", "归属感", "安全感", "价值感", "成就感", "存在感",
        # 价值判断
        "美好", "珍贵", "重要", "值得", "珍贵", "珍惜", "遗憾", "唏嘘", "感慨",
        "荒废", "沉迷", "上瘾", "内耗", "报复", "执念", "羁绊",
    })

    @classmethod
    def is_emotion_word(cls, text: str) -> bool:
        """判断是否情感/状态词（含子串匹配，应对短语变体）。

        优化：枚举文本的 2~4 字符子串查 set（O(|text|²)），
        而非遍历 80 个情感词做 `ew in text`（O(|EW| × |text|））。
        """
        text = text.strip()
        if text in cls.EMOTION_WORDS:
            return True
        # 枚举文本中 2~4 字符的连续子串，看是否命中情感词表
        tlen = len(text)
        for size in (2, 3, 4):
            if size > tlen:
                break
            for start in range(tlen - size + 1):
                if text[start:start + size] in cls.EMOTION_WORDS:
                    return True
        return False

    @staticmethod
    def specificity(text: str) -> float:
        """字符长度 specificity（仅用于 dedup 排序，不再参与 energy）。"""
        length = len(text.strip())
        if length <= 1:
            return 0.15
        if length <= 2:
            return 0.45
        if length <= 4:
            return 0.75
        return 1.0

    @staticmethod
    def _zipf_frequency(text: str) -> float:
        """查 wordfreq 的 zipf 词频（外部大语料基线），取 max(zh, en)。

        zipf 是对数词频，范围约 0-8：
          <2  极罕见（LocalSend/Syncthing）→ 专有名词
          2-4 中等（MIDI/容器配置/Docker Compose）→ 合理锚点
          >4  常见泛词（file/USB/软件/电脑）→ 该压低
        查不到返回 0（视为极罕见）。
        """
        try:
            from wordfreq import zipf_frequency
            return max(zipf_frequency(text, 'zh'), zipf_frequency(text, 'en'))
        except Exception:
            return 0.0

    @staticmethod
    def initial_energy(text: str, doc_freq: int, total_docs: int,
                       is_emotion: bool = False,
                       singleton_cap: float = 0.70) -> float:
        """计算概念节点的初始能量。

        三层设计：
          1. 个人语料 IDF：在记忆库内的稀有度（小语料区分度有限，做基线）
          2. 外部 zipf 惩罚：wordfreq 大语料词频。file/USB/软件 这种泛词
             zipf>4 被强压；LocalSend/Docker Compose zipf<2 不惩罚。
             这解决了小语料 IDF 对 df=1~2 的泛词（file df=1）无能为力的问题。
          3. 情感/状态词豁免：is_emotion=True 时跳过 zipf 惩罚——
             开心/难过虽 zipf 高（常见词），但标记个人状态，应保持高分。
          4. 单例天花板：doc_freq=1 的关键词能量不超过 singleton_cap，
             防止 keyatten 提取的复合标签（如"Docker+Compose"）以虚高能量
             通过概念图扩散把不相关笔记推到首位。

        Args:
            is_emotion: 是否情感/状态词（豁免 zipf 惩罚）。
            singleton_cap: doc_freq=1 关键词的能量上限（默认 0.70）。
        """
        idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
        idf_norm = min(1.0, idf / max(1.0, math.log(total_docs + 1) + 1.0))

        # 情感词豁免：不压 zipf
        if is_emotion:
            energy = 0.25 + 0.75 * idf_norm
        else:
            # 外部 zipf 惩罚（结合个人语料 df 保护）：
            # 核心区分：一个词在大语料高频(zipf高) + 在你语料低频(df低) → 噪声，压低
            #           一个词在大语料高频 + 在你语料也高频(df≥4) → 你的核心主题，保护
            # zipf < 2 → 不惩罚（专有名词/罕见词）
            # zipf 2~4 → 线性惩罚 0→0.5
            # zipf > 4 → 强惩罚，但如果 df≥4 说明是你的核心主题，减轻惩罚
            zipf = ConceptGraph._zipf_frequency(text)
            if zipf < 2.0:
                penalty = 1.0
            elif zipf < 4.0:
                penalty = 1.0 - 0.5 * (zipf - 2.0) / 2.0  # 2→1.0, 4→0.5
            else:
                penalty = max(0.1, 0.5 - 0.2 * (zipf - 4.0))  # 4→0.5, 5→0.3, 6→0.1
                # df 保护：在你语料里高频的核心主题词，减轻惩罚
                if doc_freq >= 4:
                    penalty = min(1.0, penalty + 0.4)
            energy = 0.25 + 0.75 * idf_norm * penalty

        # 单例天花板：df=1 的关键词只出现过一次，其概念分不应主导排序。
        # 防止 keyatten 提取的复合标签（如"Docker+Compose"）以虚高能量扩散。
        if doc_freq <= 1 and singleton_cap < 1.0:
            energy = min(energy, singleton_cap)

        return round(energy, 4)

    @staticmethod
    def dedup_concept_keywords(
        keywords: list[str],
        vectors: np.ndarray,
        threshold: float = 0.90,
    ) -> dict[str, str]:
        """余弦贪心去重近义关键词锚点，返回 原关键词 -> 规范关键词 的映射。

        合并 "服务配置" / "环境配置缺失" 这类在向量空间高度重叠、但精确串不同的
        关键词，避免三个失效模式：
          - 概念图冗余紧簇导致激活重复计数（exponential_edge_weight 给近义对
            ≥0.85 的边权，扩散时互相反复放大）
          - doc_freq 被拆碎、initial_energy / min_concept_energy 过滤失真
          - build_keyword_edges 的精确交集对近义锚点建不出边

        合并方向：specificity 更高（更长更具体）的词作为规范词，保留更多信息。
        贪心按 specificity 降序确定规范词，再把余弦 ≥ threshold 的未分配词归入。
        """
        if len(keywords) < 2:
            return {kw: kw for kw in keywords}

        sims = cosine_matrix(vectors)
        # specificity 降序（更长更具体先做规范词），文本升序保证确定性
        order = sorted(
            range(len(keywords)),
            key=lambda i: (-ConceptGraph.specificity(keywords[i]), keywords[i]),
        )
        remap: dict[str, str] = {}
        assigned = [False] * len(keywords)
        for i in order:
            if assigned[i]:
                continue
            canonical = keywords[i]
            remap[canonical] = canonical
            assigned[i] = True
            for j in order:
                if assigned[j] or j == i:
                    continue
                if float(sims[i, j]) >= threshold:
                    remap[keywords[j]] = canonical
                    assigned[j] = True
        return remap

    def add_concept(
        self,
        text: str,
        vector: np.ndarray | None,
        doc_freq: int,
        total_docs: int,
        precomputed_energy: float | None = None,
    ) -> str:
        cid = self.concept_id(text)
        if cid not in self.concepts:
            self.concepts[cid] = ConceptNode(
                id=cid,
                text=text,
                vector=vector,
                doc_freq=doc_freq,
                specificity=self.specificity(text),
                initial_energy=(precomputed_energy
                                if precomputed_energy is not None
                                else self.initial_energy(text, doc_freq, total_docs)),
            )
            self.concept_edges[cid] = {}
        elif vector is not None and self.concepts[cid].vector is None:
            self.concepts[cid].vector = vector
        return cid

    def link_event_concept(
        self, event_id: str, concept_id: str, weight: float,
        context_vector: np.ndarray | None = None,
    ) -> None:
        weight = float(max(0.0, min(1.0, weight)))
        self.event_to_concepts.setdefault(event_id, {})
        self.concept_to_events.setdefault(concept_id, {})
        old = self.event_to_concepts[event_id].get(concept_id, 0.0)
        merged = max(old, weight)
        self.event_to_concepts[event_id][concept_id] = merged
        self.concept_to_events[concept_id][event_id] = merged
        if concept_id in self.concepts:
            self.concepts[concept_id].events[event_id] = merged
        # 存储边上下文向量（查询感知路由用）
        if context_vector is not None:
            self.context_vectors[(event_id, concept_id)] = context_vector

    def link_concepts(self, source: str, target: str, weight: float) -> None:
        if source == target:
            return
        if source not in self.concepts or target not in self.concepts:
            return
        weight = float(max(0.0, min(1.0, weight)))
        self.concept_edges.setdefault(source, {})
        self.concept_edges.setdefault(target, {})
        self.concept_edges[source][target] = max(
            self.concept_edges[source].get(target, 0.0), weight
        )
        self.concept_edges[target][source] = max(
            self.concept_edges[target].get(source, 0.0), weight
        )

    def presort_neighbors(self, top_neighbors: int = 5) -> None:
        """预排序每个概念节点的 top-K 邻居，供 diffuse() 直接查表。

        在 build_concept_graph 结尾调用一次。diffuse 每个 hop 对每个
        frontier 节点做 sorted(edges)[:K] + sum(weight)，如果边结构不变
        就无需重复排序。预排序把 K 次 sort 从 O(hops × |frontier| × E log E)
        降到 O(|concepts| × E log E)。
        """
        self._sorted_neighbors = {}
        for cid, edges in self.concept_edges.items():
            if not edges:
                continue
            ranked = sorted(edges.items(), key=lambda item: item[1], reverse=True)
            ranked = ranked[:top_neighbors]
            total_weight = sum(w for _, w in ranked)
            if total_weight <= 0:
                continue
            self._sorted_neighbors[cid] = (ranked, total_weight)

    def diffuse(
        self,
        seed_concepts: dict[str, float],
        hops: int = 2,
        hop_decay: float = 0.7,
        min_activation: float = 0.05,
        top_neighbors: int = 5,
    ) -> dict[str, float]:
        """概念扩散。能量按边权归一分流，每跳衰减。"""
        activations = dict(seed_concepts)
        frontier = dict(seed_concepts)
        # 优先用预排序缓存；没有则回退到运行时排序。
        use_cache = bool(self._sorted_neighbors)

        for _ in range(hops):
            next_frontier: dict[str, float] = {}
            for cid, activation in frontier.items():
                if activation < min_activation:
                    continue
                if use_cache and cid in self._sorted_neighbors:
                    ranked, total_weight = self._sorted_neighbors[cid]
                else:
                    edges = self.concept_edges.get(cid, {})
                    if not edges:
                        continue
                    ranked = sorted(edges.items(), key=lambda item: item[1], reverse=True)
                    ranked = ranked[:top_neighbors]
                    total_weight = sum(weight for _, weight in ranked)
                    if total_weight <= 0:
                        continue
                for neighbor_id, edge_weight in ranked:
                    delta = activation * (edge_weight / total_weight) * hop_decay
                    if delta < min_activation:
                        continue
                    next_frontier[neighbor_id] = next_frontier.get(neighbor_id, 0.0) + delta
            for cid, activation in next_frontier.items():
                activations[cid] = max(activations.get(cid, 0.0), activation)
            frontier = next_frontier

        return activations

    def events_from_concepts(
        self,
        concept_activations: dict[str, float],
        query_vector: np.ndarray | None = None,
        convergence_bonus: bool = True,
        fanout_dilution: bool = True,
    ) -> dict[str, float]:
        """概念激活 → 事件得分。

        query_vector: 查询向量。如果提供且有上下文向量，启用查询感知路由：
            能量流向由 query 与边上下文向量的余弦相似度控制。
            "父亲" 节点同时连着小说事件和现实事件，但查询
            "小说里男主的爸爸怎么了" 的向量与小说边的上下文向量
            相似度 0.9、与现实边的只有 0.2，于是 90% 能量流向小说。
            无 query_vector 时回退到 fanout_dilution。
        """
        # 预计算查询向量（归一化）
        q_norm = None
        if query_vector is not None and self.context_vectors:
            q = query_vector.astype(np.float32)
            qn = np.linalg.norm(q)
            if qn > 0:
                q_norm = q / qn

        event_scores: dict[str, float] = {}
        support_count: dict[str, int] = {}
        for cid, activation in concept_activations.items():
            event_map = self.concept_to_events.get(cid, {})
            n_events = len(event_map)
            if n_events == 0:
                continue

            # ── 查询感知路由 ──
            if q_norm is not None:
                for event_id, edge_weight in event_map.items():
                    ctx = self.context_vectors.get((event_id, cid))
                    if ctx is not None:
                        # 余弦相似度（ctx 已在存入时归一化）
                        sim = float(np.dot(q_norm, ctx.astype(np.float32)))
                        gate = max(0.0, sim)  # 负相关 → 0
                    else:
                        # 没有上下文向量的边：给一个中性门控（不过滤）
                        gate = 0.5
                    event_scores[event_id] = (
                        event_scores.get(event_id, 0.0)
                        + activation * edge_weight * gate
                    )
                    support_count[event_id] = support_count.get(event_id, 0) + 1
            else:
                # ── 回退：扇出稀释（无查询向量时） ──
                dilution = (1.0 / math.sqrt(n_events)) if fanout_dilution else 1.0
                for event_id, edge_weight in event_map.items():
                    event_scores[event_id] = (
                        event_scores.get(event_id, 0.0)
                        + activation * edge_weight * dilution
                    )
                    support_count[event_id] = support_count.get(event_id, 0) + 1

        if convergence_bonus:
            for event_id, score in list(event_scores.items()):
                event_scores[event_id] = score * math.log1p(support_count[event_id])
        return event_scores

    def event_supports_from_concepts(
        self,
        concept_activations: dict[str, float],
    ) -> dict[str, list[dict]]:
        """返回事件得分由哪些概念贡献，用于 debug。"""
        supports: dict[str, list[dict]] = {}
        for cid, activation in concept_activations.items():
            concept = self.concepts.get(cid)
            if concept is None:
                continue
            for event_id, edge_weight in self.concept_to_events.get(cid, {}).items():
                contribution = activation * edge_weight
                supports.setdefault(event_id, []).append({
                    "concept_id": cid,
                    "concept": concept.text,
                    "activation": round(float(activation), 4),
                    "edge_weight": round(float(edge_weight), 4),
                    "contribution": round(float(contribution), 4),
                })
        for event_id in list(supports.keys()):
            supports[event_id].sort(
                key=lambda item: item["contribution"],
                reverse=True,
            )
        return supports

    # ─── 持久化 ───────────────────────────────────────────────

    def save(self, directory) -> None:
        """把概念图写入目录。

        产物：
          - concept_graph.json：概念节点元数据 + 三张邻接表的边列表
          - concept_vectors.npz：概念向量矩阵 + 边上下文向量矩阵

        _sorted_neighbors 不持久化，load() 后由 presort_neighbors() 重建。
        空 store（无概念）只写 json，不写 npz，load 时静默跳过。
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # 概念元数据 + events 字典（不含向量，向量进 npz）
        concept_meta = []
        concept_order = []  # 对齐 npz 的 concept_vec 矩阵
        for cid, concept in self.concepts.items():
            concept_order.append(cid)
            concept_meta.append({
                "id": concept.id,
                "text": concept.text,
                "initial_energy": float(concept.initial_energy),
                "doc_freq": int(concept.doc_freq),
                "specificity": float(concept.specificity),
                "events": {eid: float(w) for eid, w in concept.events.items()},
            })

        # 三张邻接表压平成 [src, tgt, weight] 边列表（json 友好）
        concept_edges_list = [
            [src, tgt, float(w)]
            for src, nbrs in self.concept_edges.items()
            for tgt, w in nbrs.items()
        ]
        # concept_to_events 是 event_to_concepts 的反向视图，只存一份
        event_concept_list = [
            [eid, cid, float(w)]
            for eid, cmap in self.event_to_concepts.items()
            for cid, w in cmap.items()
        ]

        payload = {
            "concepts": concept_meta,
            "concept_edges": concept_edges_list,
            "event_to_concepts": event_concept_list,
        }
        with open(path / "concept_graph.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        # 向量：概念向量 + 边上下文向量
        vectors = {}
        concept_vecs = [
            self.concepts[cid].vector
            for cid in concept_order
            if self.concepts[cid].vector is not None
        ]
        if concept_vecs:
            vectors["concept_ids"] = np.array(concept_order, dtype=object)
            vectors["concept_vec"] = np.vstack(concept_vecs).astype(np.float32)

        if self.context_vectors:
            ctx_keys = list(self.context_vectors.keys())
            # 用两个并行数组存 tuple key，避免 structured array 跨版本兼容问题
            vectors["ctx_event_id"] = np.array([k[0] for k in ctx_keys], dtype=object)
            vectors["ctx_concept_id"] = np.array([k[1] for k in ctx_keys], dtype=object)
            vectors["ctx_vec"] = np.vstack(
                [self.context_vectors[k] for k in ctx_keys]
            ).astype(np.float32)

        if vectors:
            np.savez(path / "concept_vectors.npz", **vectors)

    def load(self, directory, top_neighbors: int = 5) -> bool:
        """从目录加载概念图。

        若目录无 concept_graph.json 视为旧 store，返回 False（静默跳过，
        保持向后兼容）。加载后重建 _sorted_neighbors 缓存。
        """
        path = Path(directory)
        json_path = path / "concept_graph.json"
        if not json_path.exists():
            return False

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # 重置状态
        self.concepts = {}
        self.event_to_concepts = {}
        self.concept_to_events = {}
        self.concept_edges = {}
        self.context_vectors = {}
        self._sorted_neighbors = {}

        # 先加载向量矩阵（若有），后面回填到概念节点
        concept_vec_map: dict[str, np.ndarray] = {}
        npz_path = path / "concept_vectors.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as data:
                if "concept_ids" in data:
                    cids = [str(x) for x in data["concept_ids"]]
                    cvecs = data["concept_vec"]
                    concept_vec_map = dict(zip(cids, cvecs))
                if "ctx_event_id" in data:
                    ev_ids = [str(x) for x in data["ctx_event_id"]]
                    co_ids = [str(x) for x in data["ctx_concept_id"]]
                    cvecs = data["ctx_vec"]
                    for eid, cid, vec in zip(ev_ids, co_ids, cvecs):
                        self.context_vectors[(eid, cid)] = vec.astype(np.float32)

        # 恢复概念节点
        for cm in payload.get("concepts", []):
            cid = cm["id"]
            node = ConceptNode(
                id=cid,
                text=cm["text"],
                vector=concept_vec_map.get(cid),
                initial_energy=float(cm["initial_energy"]),
                doc_freq=int(cm["doc_freq"]),
                specificity=float(cm["specificity"]),
                events={eid: float(w) for eid, w in cm.get("events", {}).items()},
            )
            self.concepts[cid] = node
            self.concept_edges[cid] = {}

        # 恢复邻接表
        for src, tgt, w in payload.get("concept_edges", []):
            self.concept_edges.setdefault(src, {})
            self.concept_edges.setdefault(tgt, {})
            self.concept_edges[src][tgt] = float(w)
            self.concept_edges[tgt][src] = float(w)

        for eid, cid, w in payload.get("event_to_concepts", []):
            self.event_to_concepts.setdefault(eid, {})[cid] = float(w)
            self.concept_to_events.setdefault(cid, {})[eid] = float(w)

        # 重建预排序缓存
        self.presort_neighbors(top_neighbors=top_neighbors)
        return True


def exponential_edge_weight(
    similarity: float,
    threshold: float = 0.68,
    temperature: float = 0.08,
) -> float:
    """线性距离输入，指数级区分输出。"""
    if similarity < threshold:
        return 0.0
    raw = math.exp((similarity - threshold) / max(temperature, 1e-6))
    # sim=threshold -> 0，sim 越接近 1 越接近 1。
    max_raw = math.exp((1.0 - threshold) / max(temperature, 1e-6))
    return (raw - 1.0) / max(max_raw - 1.0, 1e-6)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vecs = vectors.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return vecs @ vecs.T
