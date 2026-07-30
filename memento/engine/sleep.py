"""
睡眠巩固模块 - 离线巩固与创造

睡眠中的行为（按顺序执行）：
1. 回放与巩固：序列回放近期高 λ 节点
2. 漫游联想：随机游走强化常用路径
3. 探索建边：高 λ 节点主动探索新连接
4. LLM 裁决建边：高价值节点的近邻候选交 LLM 裁决（可选，严格限定）
5. 节点融合：近乎重复的节点合并成融合节点（可选）
6. 聚类凝聚：高度互联节点簇 → 聚合节点
7. 遗忘与修剪：全局衰减 + 修剪弱边

阶段 4/5 是离线、低频、有缓存的可选增强，不进 add/query 热路径。
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from memento.graph.memory_graph import MemoryGraph
from memento.index.vector_index import VectorIndex
from memento.engine.decay import DecayEngine


@dataclass
class SleepReport:
    """睡眠周期报告"""
    edges_strengthened: int = 0
    explore_edges_created: int = 0
    edges_pruned: int = 0
    nodes_dormant: int = 0
    cluster_nodes_created: int = 0
    replay_count: int = 0
    walk_count: int = 0
    # LLM 裁决建边
    llm_edges_created: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_candidates: int = 0
    # 节点融合
    fusions_created: int = 0
    nodes_fused: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  Memento 睡眠报告",
            "=" * 50,
            f"  回放巩固:       {self.replay_count} 个节点参与序列回放",
            f"  边强化:         {self.edges_strengthened} 条边被增强",
            f"  漫游联想:       {self.walk_count} 次随机游走",
            f"  探索建边:       {self.explore_edges_created} 条新探索边",
            f"  LLM 裁决建边:   {self.llm_edges_created} 条 "
            f"(候选 {self.llm_candidates}, 调用 {self.llm_calls}, "
            f"缓存命中 {self.llm_cache_hits})",
            f"  节点融合:       {self.fusions_created} 个融合节点 "
            f"({self.nodes_fused} 个源节点被融合)",
            f"  聚类凝聚:       {self.cluster_nodes_created} 个聚合节点",
            f"  遗忘修剪:       {self.edges_pruned} 条边被修剪",
            f"  节点休眠:       {self.nodes_dormant} 个节点进入休眠/冷存储",
            "=" * 50,
        ]
        return "\n".join(lines)


class SleepEngine:
    """睡眠巩固引擎"""

    def __init__(self, graph: MemoryGraph, vector_index: VectorIndex,
                 decay_engine: DecayEngine,
                 explore_kappa: float = 0.05,       # 探索系数 κ
                 explore_top_pct: float = 0.10,     # 探索节点比例
                 cluster_min_size: int = 5,          # 聚类最小节点数
                 cluster_min_weight: float = 0.3,    # 聚类最小内部边强度
                 walk_hops: int = 4,                 # 漫游跳数
                 walk_count: int = 10,               # 漫游次数
                 # LLM 裁决建边（严格限定：仅此处可调 LLM）
                 llm_enabled: bool = False,
                 llm_curate_cosine: float = 0.35,    # 预筛余弦阈值（比建边宽松）
                 llm_curate_top_k: int = 8,          # 每节点候选数
                 llm_max_calls_per_cycle: int = 50,  # 每周期 LLM 调用硬上限
                 llm_client: object = None,
                 # 节点融合
                 fusion_cosine_threshold: float = 0.85,  # 融合预筛余弦（严）
                 fusion_k: int = 5,
                 fusion_max_per_cycle: int = 20,
                 enable_fusion: bool = False):        # 融合每周期上限
        self.graph = graph
        self.vector_index = vector_index
        self.decay_engine = decay_engine
        self.explore_kappa = explore_kappa
        self.explore_top_pct = explore_top_pct
        self.cluster_min_size = cluster_min_size
        self.cluster_min_weight = cluster_min_weight
        self.walk_hops = walk_hops
        self.walk_count = walk_count
        self._cluster_counter = 0
        # LLM 巩固参数
        self.llm_enabled = llm_enabled
        self.llm_curate_cosine = llm_curate_cosine
        self.llm_curate_top_k = llm_curate_top_k
        self.llm_max_calls_per_cycle = llm_max_calls_per_cycle
        self._llm_client = llm_client
        # 融合参数
        self.fusion_cosine_threshold = fusion_cosine_threshold
        self.fusion_k = fusion_k
        self.fusion_max_per_cycle = fusion_max_per_cycle
        self.enable_fusion = enable_fusion
        self._fusion_counter = 0

    def run_sleep_cycle(self) -> SleepReport:
        """
        执行完整睡眠周期

        顺序：回放 → 漫游 → 探索 → LLM裁决建边 → 节点融合 → 聚类 → 衰减/修剪
        """
        report = SleepReport()

        # 1. 回放与巩固
        self._replay_and_consolidate(report)

        # 2. 漫游联想
        self._random_walk(report)

        # 3. 探索建边（算法层，保覆盖）
        self._explore_edges(report)

        # 4. LLM 裁决建边（可选，叠加在算法层之上）
        if self.llm_enabled:
            self._llm_curate_edges(report)

        # 5. 节点融合（可选）
        if self.enable_fusion:
            self._fuse_duplicates(report)

        # 6. 聚类凝聚
        self._cluster_consolidation(report)

        # 7. 遗忘与修剪（执行一个时钟步衰减）
        decay_stats = self.decay_engine.step()
        report.edges_pruned = decay_stats.get("edges_pruned", 0)
        report.nodes_dormant = (decay_stats.get("nodes_dormant", 0)
                                + decay_stats.get("nodes_cold", 0))

        return report

    # ── 10.1 回放与巩固 ────────────────────────────────────

    def _replay_and_consolidate(self, report: SleepReport):
        """选取近期生命力最高的节点，序列回放强化相邻边"""
        active_nodes = [n for n in self.graph.nodes.values()
                        if n.status == "active"]
        if not active_nodes:
            return

        # Top 5% 高生命力节点
        top_k = max(5, int(len(active_nodes) * 0.05))
        top_nodes = sorted(active_nodes, key=lambda n: n.vitality,
                           reverse=True)[:top_k]

        # 按 access_count 排序（模拟最近激活顺序）
        replay_sequence = sorted(top_nodes, key=lambda n: n.access_count,
                                 reverse=True)
        report.replay_count = len(replay_sequence)

        # 序列中相邻节点之间增强边
        for i in range(len(replay_sequence)):
            for j in range(i + 1, min(i + 4, len(replay_sequence))):
                n_a, n_b = replay_sequence[i], replay_sequence[j]
                edge = self.graph.get_edge(n_a.id, n_b.id)
                if edge:
                    edge.weight = min(1.0, edge.weight + 0.03)
                    report.edges_strengthened += 1
                else:
                    # 回放序列中的新关联 → 弱共现边
                    self.graph.add_edge(n_a.id, n_b.id, weight=0.02,
                                        edge_type="cooccurrence")
                    report.edges_strengthened += 1

    # ── 10.2 漫游联想 ──────────────────────────────────────

    def _random_walk(self, report: SleepReport):
        """从高 λ 节点出发，沿边随机游走，强化路径"""
        active_nodes = [n for n in self.graph.nodes.values()
                        if n.status == "active" and n.vitality > 0.3]
        if not active_nodes:
            return

        for _ in range(self.walk_count):
            start = random.choice(active_nodes)
            current_id = start.id
            path = [current_id]

            for _ in range(self.walk_hops):
                neighbors = self.graph.get_neighbors(current_id)
                if not neighbors:
                    break

                # 按边强度加权随机选择
                n_ids = list(neighbors.keys())
                weights = [neighbors[nid].weight for nid in n_ids]
                total_w = sum(weights)
                if total_w <= 0:
                    break
                probs = [w / total_w for w in weights]

                next_id = random.choices(n_ids, weights=probs, k=1)[0]
                path.append(next_id)

                # 强化走过的边
                edge = self.graph.get_edge(current_id, next_id)
                if edge:
                    edge.weight = min(1.0, edge.weight + 0.01)
                    report.edges_strengthened += 1

                current_id = next_id

            if len(path) > 1:
                report.walk_count += 1

    # ── 10.3 探索建边 ──────────────────────────────────────

    def _explore_edges(self, report: SleepReport):
        """高 λ 节点主动探索向量空间中的新连接"""
        active_nodes = [n for n in self.graph.nodes.values()
                        if n.status == "active"]
        if not active_nodes:
            return

        # 选取高生命力节点
        threshold = sorted([n.vitality for n in active_nodes],
                           reverse=True)[
            :max(1, int(len(active_nodes) * self.explore_top_pct))
        ]
        min_vitality = threshold[-1] if threshold else 0.5
        high_lambda = [n for n in active_nodes
                       if n.vitality >= min_vitality]

        if self.vector_index.size == 0:
            return

        for node in high_lambda[:20]:  # 最多 20 个探索者
            if node.vector is None:
                continue

            neighbors_set = set(self.graph.get_neighbors(node.id).keys())
            candidates = self.vector_index.search(node.vector, k=10)

            for cand_id, sim in candidates:
                if cand_id == node.id or cand_id in neighbors_set:
                    continue
                cand_node = self.graph.get_node(cand_id)
                if not cand_node or cand_node.status != "active":
                    continue

                # w_explore = κ × λ_i × λ_j
                w_explore = (self.explore_kappa
                             * node.vitality * cand_node.vitality)
                if w_explore > 0.001:
                    self.graph.add_edge(node.id, cand_id,
                                        weight=w_explore,
                                        edge_type="explore")
                    report.explore_edges_created += 1
                break  # 每个节点只探索一条新边

    # ── 10.4 LLM 裁决建边 ────────────────────────────────

    def _get_llm_client(self):
        """延迟初始化 LLM 客户端。不可用返回 None。"""
        if self._llm_client is None:
            from memento.engine.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client if self._llm_client.available else None

    def _select_high_value_nodes(self, max_count: int = 20) -> list:
        """选取高价值节点（复用 _explore_edges 的 top-pct-by-vitality 逻辑）。"""
        active_nodes = [n for n in self.graph.nodes.values()
                        if n.status == "active" and n.vector is not None]
        if not active_nodes:
            return []
        threshold = sorted([n.vitality for n in active_nodes], reverse=True)[
            :max(1, int(len(active_nodes) * self.explore_top_pct))
        ]
        min_vitality = threshold[-1] if threshold else 0.5
        high_value = [n for n in active_nodes if n.vitality >= min_vitality]
        return high_value[:max_count]

    def _llm_curate_edges(self, report: SleepReport):
        """高价值节点的近邻候选交 LLM 裁决建边（叠加在算法层之上）。

        严格限定：
          - 仅对余弦 > llm_curate_cosine 的候选对调用 LLM
          - 每周期 llm_max_calls_per_cycle 硬上限
          - (text_a, text_b) hash 缓存，同对永不重判
          - LLM 不可用时跳过，不阻断睡眠
        """
        client = self._get_llm_client()
        if client is None:
            return
        if self.vector_index.size == 0:
            return

        high_value = self._select_high_value_nodes()
        # 去重候选对（无向，避免 O(n²) 重复）
        candidate_pairs: list[tuple] = []  # (node_a, node_b, cosine)
        seen_pairs: set[tuple[str, str]] = set()
        for node in high_value:
            neighbors_set = set(self.graph.get_neighbors(node.id).keys())
            candidates = self.vector_index.search(
                node.vector, k=self.llm_curate_top_k + 1
            )
            for cand_id, sim in candidates:
                if cand_id == node.id or sim < self.llm_curate_cosine:
                    continue
                if cand_id in neighbors_set:
                    continue
                cand_node = self.graph.get_node(cand_id)
                if not cand_node or cand_node.status != "active":
                    continue
                pair = tuple(sorted([node.id, cand_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                a = node if node.id == pair[0] else cand_node
                b = cand_node if a is node else node
                candidate_pairs.append((a, b, sim))

        report.llm_candidates = len(candidate_pairs)
        calls_budget = self.llm_max_calls_per_cycle
        for node_a, node_b, sim in candidate_pairs:
            if client.calls >= calls_budget:
                break
            verdict = client.judge_relation(node_a.text, node_b.text, cosine=sim)
            if verdict.verdict == "link":
                weight = max(verdict.weight, float(sim))
                weight = min(1.0, max(0.0, weight))
                self.graph.add_edge(node_a.id, node_b.id,
                                    weight=weight, edge_type="llm")
                report.llm_edges_created += 1
            # verdict == "merge" 在此阶段不处理，留给 _fuse_duplicates
            # （_fuse_duplicates 有更严的余弦阈值）

        report.llm_calls = client.calls
        report.llm_cache_hits = client.cache_hits

    # ── 10.5 节点融合 ──────────────────────────────────────

    def _fuse_duplicates(self, report: SleepReport):
        """近乎重复的节点合并成融合节点。

        检测：对 active 节点向量搜 top-k，取余弦 > fusion_cosine_threshold
        的候选对。裁决：候选对送 LLM 判定是否同一件事。融合：新建融合节点
        （LLM 合成文本 + 加权平均向量），源节点标 status="superseded"
        + superseded_by 指向融合节点（自动从检索消失）。
        """
        if self.vector_index.size == 0:
            return
        client = self._get_llm_client()
        # 融合必须有 LLM（否则纯余弦判同太危险）；LLM 不可用则跳过
        if client is None:
            return

        active_nodes = [n for n in self.graph.nodes.values()
                        if n.status == "active" and n.vector is not None
                        and "__fusion__" not in n.tags]
        # 收集候选对（无向，去重）
        candidate_pairs: list[tuple] = []
        seen_pairs: set[tuple[str, str]] = set()
        for node in active_nodes:
            candidates = self.vector_index.search(node.vector, k=self.fusion_k + 1)
            for cand_id, sim in candidates:
                if cand_id == node.id or sim < self.fusion_cosine_threshold:
                    continue
                pair = tuple(sorted([node.id, cand_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                a = self.graph.get_node(pair[0])
                b = self.graph.get_node(pair[1])
                if a is None or b is None:
                    continue
                if a.status != "active" or b.status != "active":
                    continue
                candidate_pairs.append((a, b, sim))

        # 按余弦降序处理（最像的先融合）
        candidate_pairs.sort(key=lambda x: x[2], reverse=True)
        fusions_this_cycle = 0
        consumed: set[str] = set()  # 已被融合的节点不再参与

        for node_a, node_b, sim in candidate_pairs:
            if fusions_this_cycle >= self.fusion_max_per_cycle:
                break
            if node_a.id in consumed or node_b.id in consumed:
                continue
            verdict = client.judge_relation(node_a.text, node_b.text, cosine=sim)
            if verdict.verdict != "merge":
                continue
            fusion_id = self._create_fusion_node({node_a.id, node_b.id})
            if fusion_id is None:
                continue
            # 源节点标记 superseded（检索自动过滤 active；decay 不再改其状态）
            for src_id in (node_a.id, node_b.id):
                src = self.graph.get_node(src_id)
                if src:
                    src.status = "superseded"
                    src.superseded_by = fusion_id
                    consumed.add(src_id)
            report.fusions_created += 1
            report.nodes_fused += 2
            fusions_this_cycle += 1

    def _create_fusion_node(self, src_ids: set) -> str | None:
        """创建融合节点。复用 _create_cluster_node 的向量/ω 策略，但文本用
        源节点原文的对话流拼接（Q1→A1→Q2→A2），不做 LLM 概括——保留原汁
        味，零信息损失。标记 __fusion__ / sleep_fusion。

        源节点的边迁移到融合节点（避免悬空）。源节点本身的 status 由调用方
        设为 superseded。
        """
        import numpy as np

        # 按源 id 排序，保证对话流顺序稳定（qa_0013 在 qa_0014 前）
        sorted_ids = sorted(src_ids)
        nodes = [self.graph.get_node(nid) for nid in sorted_ids
                 if self.graph.get_node(nid)]
        if len(nodes) < 2:
            return None

        vectors = np.array([n.vector for n in nodes if n.vector is not None])
        if len(vectors) < 2:
            return None
        weights = np.array([n.importance for n in nodes if n.vector is not None])
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        avg_vector = np.average(vectors, axis=0, weights=weights)

        # ω = 簇内最大值；λ = 源节点最大值（融合节点应高活跃）
        max_omega = max(n.importance for n in nodes)
        max_lambda = max(n.vitality for n in nodes)

        # 对话流拼接：保留原文，连续的 Q/A 块用换行分隔
        fused_text = "\n".join(n.text.rstrip() for n in nodes)

        self._fusion_counter += 1
        fusion_id = f"fusion_{self._fusion_counter:04d}"

        from memento.models import Node
        fusion_node = Node(
            id=fusion_id,
            text=fused_text,
            vector=avg_vector,
            importance=max_omega,
            vitality=max_lambda,
            tags=["__fusion__"],
            source="sleep_fusion",
            status="active",
            fused_from=sorted_ids,
        )
        self.graph.add_node(fusion_node)
        self.vector_index.add(fusion_id, avg_vector)

        # 源节点的边迁移到融合节点
        for src_id in src_ids:
            neighbors = self.graph.get_neighbors(src_id)
            for nbr_id, edge in neighbors.items():
                if nbr_id in src_ids:
                    continue
                self.graph.add_edge(fusion_id, nbr_id,
                                    weight=edge.weight,
                                    edge_type=edge.edge_type)
        return fusion_id

    # ── 10.6 聚类凝聚 ──────────────────────────────────────

    def _cluster_consolidation(self, report: SleepReport):
        """
        检测高度互联的节点簇，生成聚合节点

        使用简单的贪心社区检测：
        - 从度数最高的节点开始
        - 扩展社区：加入与当前社区平均边强度 > 阈值的邻居
        """
        visited: Set[str] = set()
        active_nodes = [n for n in self.graph.nodes.values()
                        if n.status == "active" and "__fusion__" not in n.tags]

        # 按度数降序
        sorted_by_degree = sorted(
            active_nodes,
            key=lambda n: len(self.graph.get_neighbors(n.id)),
            reverse=True
        )

        for seed in sorted_by_degree:
            if seed.id in visited:
                continue

            neighbors = self.graph.get_neighbors(seed.id)
            if len(neighbors) < self.cluster_min_size - 1:
                continue

            # 贪心扩展社区
            community = {seed.id}
            candidates = set(neighbors.keys())

            while candidates:
                best_id = None
                best_avg_w = 0

                for cand_id in candidates:
                    if cand_id in visited:
                        continue
                    cand = self.graph.get_node(cand_id)
                    # 跳过：非 active（superseded/dormant/cold 源节点不进社区）
                    # 跳过：已是融合/聚合节点（派生节点不进别人的社区，避免重叠）
                    if (cand is None or cand.status != "active"
                            or "__fusion__" in cand.tags
                            or "__cluster__" in cand.tags):
                        continue
                    cand_neighbors = self.graph.get_neighbors(cand_id)
                    overlap = community & set(cand_neighbors.keys())
                    if not overlap:
                        continue
                    avg_w = (sum(cand_neighbors[cid].weight for cid in overlap)
                             / len(community))
                    if avg_w > self.cluster_min_weight and avg_w > best_avg_w:
                        best_avg_w = avg_w
                        best_id = cand_id

                if best_id is None:
                    break
                community.add(best_id)
                new_cands = set(self.graph.get_neighbors(best_id).keys())
                candidates = (candidates | new_cands) - community

            if len(community) >= self.cluster_min_size:
                self._create_cluster_node(community)
                visited.update(community)
                report.cluster_nodes_created += 1

    def _create_cluster_node(self, node_ids: set):
        """创建聚合节点"""
        import numpy as np

        nodes = [self.graph.get_node(nid) for nid in node_ids
                 if self.graph.get_node(nid)]
        if len(nodes) < self.cluster_min_size:
            return

        # text = 所有子节点文本的拼接摘要
        texts = [n.text for n in nodes]
        combined = " | ".join(texts[:8])
        if len(texts) > 8:
            combined += f" | ... (共 {len(texts)} 条)"

        # vector = 加权平均
        vectors = np.array([n.vector for n in nodes if n.vector is not None])
        if len(vectors) == 0:
            return
        weights = np.array([n.importance for n in nodes
                            if n.vector is not None])
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        avg_vector = np.average(vectors, axis=0, weights=weights)

        # ω = 簇内最大值
        max_omega = max(n.importance for n in nodes)

        self._cluster_counter += 1
        cluster_id = f"cluster_{self._cluster_counter:04d}"

        from memento.models import Node
        cluster_node = Node(
            id=cluster_id,
            text=f"[聚合] {combined}",
            vector=avg_vector,
            importance=max_omega,
            vitality=0.8,
            tags=["__cluster__"],
            source="sleep_cluster",
            status="active",
        )

        self.graph.add_node(cluster_node)
        self.vector_index.add(cluster_id, avg_vector)

        # 与所有子节点建立强边
        for nid in node_ids:
            self.graph.add_edge(cluster_id, nid, weight=0.8,
                                edge_type="manual")
