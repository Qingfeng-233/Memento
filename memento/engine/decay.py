"""
时间衰减与动态系统

一切时间流逝通过系统"时钟步"体现，不依赖真实日期。
一个时钟步 = 一次用户输入 / 一次睡眠周期 / N 次内部操作。

动态属性：
  - 生命力 λ: 瞬时活跃度，自然衰减，使用则提升
  - 重要性 ω: 长期权重，手动标记或自动沉淀（结构中心性）
  - 边强度 w: 使用强化，自然衰减，低强度修剪
"""

from memento.graph.memory_graph import MemoryGraph


class DecayEngine:
    """时间衰减引擎 - 管理 λ、ω、w 的动态更新"""

    def __init__(self, graph: MemoryGraph,
                 decay_rate: float = 0.01,     # λ 基础衰减率
                 edge_decay: float = 0.005,    # w 基础衰减率
                 w_min: float = 0.01,          # 边修剪阈值
                 lambda_min: float = 0.05,     # λ 冷存储阈值
                 omega_threshold: float = 0.8, # ω 保护阈值（λ 下限保护）
                 lambda_floor: float = 0.3,    # ω 保护时的 λ 下限
                 mu: float = 0.001):           # ω 结构沉淀速率
        self.graph = graph
        self.decay_rate = decay_rate
        self.edge_decay = edge_decay
        self.w_min = w_min
        self.lambda_min = lambda_min
        self.omega_threshold = omega_threshold
        self.lambda_floor = lambda_floor
        self.mu = mu

    def step(self) -> dict:
        """
        执行一个时钟步的衰减

        Returns:
            衰减统计信息
        """
        stats = {
            "edges_pruned": 0,
            "nodes_dormant": 0,
            "nodes_cold": 0,
            "omega_increased": 0,
        }

        # ── 1. 生命力 λ 衰减 ──
        # λ_i ← λ_i × (1 - decay_rate × (1 - ω_i))
        for node in self.graph.nodes.values():
            decay = self.decay_rate * (1 - node.importance)
            node.vitality *= (1 - decay)

            # 保护机制：ω 极高时 λ 有下限
            if (node.importance >= self.omega_threshold
                    and node.vitality < self.lambda_floor):
                node.vitality = self.lambda_floor

        # ── 2. 边强度 w 衰减 + 修剪 ──
        edges_to_remove = []
        for src, tgt, edge in self.graph.get_all_edges():
            src_node = self.graph.get_node(src)
            tgt_node = self.graph.get_node(tgt)

            rate = self.edge_decay
            # 若任一端 ω 高，衰减速率减半
            if (src_node and src_node.importance >= self.omega_threshold) or \
               (tgt_node and tgt_node.importance >= self.omega_threshold):
                rate *= 0.5

            # 探索边衰减速度为普通边的 3 倍
            # LLM 裁决边(llm)、融合血缘边(manual)质量高，用普通速率衰减，不加速
            if edge.edge_type == "explore":
                rate *= 3.0

            edge.weight *= (1 - rate)

            # 修剪弱边
            if edge.weight < self.w_min:
                edges_to_remove.append((src, tgt))

        for src, tgt in edges_to_remove:
            self._remove_edge(src, tgt)
            stats["edges_pruned"] += 1

        # ── 3. 节点状态更新 ──
        for node in self.graph.nodes.values():
            if node.vitality < self.lambda_min and node.importance < 0.3:
                if node.status != "cold":
                    node.status = "cold"
                    stats["nodes_cold"] += 1
            elif node.vitality < self.lambda_min * 2 and node.importance < 0.5:
                if node.status == "active":
                    node.status = "dormant"
                    stats["nodes_dormant"] += 1

        # ── 4. 结构中心性 → ω 沉淀 ──
        # Δω_i = μ × (Σ w_ji / max_degree) × (1 - ω_i)
        max_degree = max(
            (len(self.graph.get_neighbors(nid)) for nid in self.graph.nodes),
            default=1
        )
        if max_degree == 0:
            max_degree = 1

        for node_id, node in self.graph.nodes.items():
            neighbors = self.graph.get_neighbors(node_id)
            if not neighbors:
                continue
            total_incoming_weight = sum(e.weight for e in neighbors.values())
            delta_omega = (self.mu
                           * (total_incoming_weight / max_degree)
                           * (1 - node.importance))
            if delta_omega > 1e-6:
                node.importance = min(1.0, node.importance + delta_omega)
                stats["omega_increased"] += 1

        return stats

    def boost_vitality(self, node_id: str, boost: float = 0.2):
        """提升节点生命力"""
        node = self.graph.get_node(node_id)
        if node:
            node.vitality = min(1.0, node.vitality + boost)

    def _remove_edge(self, src: str, tgt: str):
        """从图中移除一条边"""
        if src in self.graph._adjacency:
            self.graph._adjacency[src].pop(tgt, None)
        if tgt in self.graph._adjacency:
            self.graph._adjacency[tgt].pop(src, None)
        # 更新边计数
        src_node = self.graph.get_node(src)
        tgt_node = self.graph.get_node(tgt)
        if src_node:
            src_node.edge_count = len(self.graph.get_neighbors(src))
        if tgt_node:
            tgt_node.edge_count = len(self.graph.get_neighbors(tgt))
