"""
激活扩散引擎 - 搜索 / 联想的核心

当系统接收查询时：
1. 种子获取：RAG 找到最相似的 k 个节点
2. 扩散传播：能量沿边传播 2~3 跳
3. 累积与过滤：去除低激活节点
4. 结果排序：重要性放大输出
5. 使用强化：被想起的记忆更容易再被想起
"""

from typing import List, Tuple, Dict, Set
from memento.graph.memory_graph import MemoryGraph
from memento.index.vector_index import VectorIndex


class DiffusionEngine:
    """
    激活扩散引擎

    公式：
        种子激活: a_i = sim(q, v_i) × (1 + α·ω_i)
        扩散传播: Δa_j = a_i × w_ij × β × (1 + γ·ω_i) × λ_i
        最终得分: score_j = a_j × (1 + δ·ω_j)
    """

    def __init__(self, graph: MemoryGraph, vector_index: VectorIndex,
                 alpha: float = 0.3,     # 重要性对种子激活的放大系数
                 beta: float = 0.6,      # 全局扩散衰减因子
                 gamma: float = 0.2,     # 源节点重要性对传播的放大系数
                 delta: float = 0.3,     # 重要性对最终得分的放大系数
                 epsilon: float = 0.01,  # 激活值过滤阈值
                 hops: int = 2,          # 扩散跳数
                 vitality_boost: float = 0.2,   # 命中节点的生命力提升
                 edge_reinforce: float = 0.02): # 命中边的强度增强
        self.graph = graph
        self.vector_index = vector_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.hops = hops
        self.vitality_boost = vitality_boost
        self.edge_reinforce = edge_reinforce

    def query(self, query_vector, k: int = 10,
              seed_k: int = 20) -> List[Tuple[str, float]]:
        """
        执行查询：RAG 种子检索 + 扩散联想

        Args:
            query_vector: 查询向量
            k: 最终返回数量
            seed_k: RAG 种子候选数量

        Returns:
            [(node_id, final_score), ...] 按得分降序
        """
        # ── Step 1: 种子获取 ──
        seeds = self.vector_index.search(query_vector, k=seed_k)
        if not seeds:
            return []

        # 初始化激活值
        activations: Dict[str, float] = {}
        for node_id, sim_score in seeds:
            node = self.graph.get_node(node_id)
            if node and node.status == "active":
                # a_i = sim(q, v_i) × (1 + α·ω_i)
                a_i = max(0, sim_score) * (1 + self.alpha * node.importance)
                activations[node_id] = a_i

        if not activations:
            return []

        # ── Step 2: 扩散传播（多跳） ──
        for hop in range(self.hops):
            new_activations: Dict[str, float] = {}
            # 只传播当前跳中有激活值的节点
            active_nodes = set(activations.keys())
            processed: Set[str] = set()

            for node_id, a_i in activations.items():
                if a_i < self.epsilon:
                    continue
                node = self.graph.get_node(node_id)
                if not node:
                    continue

                neighbors = self.graph.get_neighbors(node_id)
                for neighbor_id, edge in neighbors.items():
                    if neighbor_id in processed:
                        continue
                    neighbor = self.graph.get_node(neighbor_id)
                    if not neighbor or neighbor.status != "active":
                        continue

                    # Δa_j = a_i × w_ij × β × (1 + γ·ω_i) × λ_i
                    delta_a = (a_i * edge.weight * self.beta
                               * (1 + self.gamma * node.importance)
                               * node.vitality)

                    # 逐跳衰减：第 2 跳、第 3 跳能量更少
                    delta_a *= (0.7 ** hop)

                    if neighbor_id not in new_activations:
                        new_activations[neighbor_id] = 0
                    new_activations[neighbor_id] += delta_a

                processed.add(node_id)

            # 累积新激活值
            for nid, val in new_activations.items():
                if nid not in activations:
                    activations[nid] = val
                else:
                    activations[nid] += val

        # ── Step 3: 过滤 ──
        activations = {nid: a for nid, a in activations.items()
                       if a >= self.epsilon}

        # ── Step 4: 结果排序 ──
        scored_results = []
        for node_id, a_j in activations.items():
            node = self.graph.get_node(node_id)
            if node:
                # final_score = a_j × (1 + δ·ω_j)
                final_score = a_j * (1 + self.delta * node.importance)
                scored_results.append((node_id, final_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        final_results = scored_results[:k]

        # ── Step 5: 使用强化（副作用） ──
        self._reinforce(final_results)

        return final_results

    def _reinforce(self, results: List[Tuple[str, float]]):
        """
        使用强化：被想起的记忆更容易再被想起

        - 命中节点：λ += boost
        - 命中边：  w += η
        """
        result_ids = [nid for nid, _ in results]
        result_set = set(result_ids)

        for node_id in result_ids:
            node = self.graph.get_node(node_id)
            if node:
                node.vitality = min(1.0, node.vitality + self.vitality_boost)
                node.access_count += 1

        # 强化结果节点之间的边
        for i, nid_a in enumerate(result_ids):
            neighbors = self.graph.get_neighbors(nid_a)
            for nid_b in result_ids[i + 1:]:
                if nid_b in neighbors:
                    edge = neighbors[nid_b]
                    edge.weight = min(1.0, edge.weight + self.edge_reinforce)
