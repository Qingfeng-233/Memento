"""
记忆图 - 记忆 B：节点图联想网络

节点通过边连接，边即经验。
边的建立基于情境共现、主动关联或自主探索。
"""

from typing import Dict, Set, Tuple, List
from memento.models import Node, Edge


class MemoryGraph:
    """记忆关联图，管理节点间的经验连接"""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        # 邻接表：node_id -> {neighbor_id -> Edge}
        self._adjacency: Dict[str, Dict[str, Edge]] = {}

    # ─── 节点管理 ───────────────────────────────────────────

    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = {}

    def get_node(self, node_id: str) -> Node:
        return self.nodes.get(node_id)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    # ─── 边管理 ─────────────────────────────────────────────

    def add_edge(self, source: str, target: str, weight: float = 0.1,
                 edge_type: str = "cooccurrence"):
        """
        添加或增强边（无向边）

        如果边已存在，累加权重：w_ij += delta_w
        """
        if source == target:
            return
        if source not in self._adjacency or target not in self._adjacency:
            return

        key = tuple(sorted([source, target]))
        src, tgt = key

        if tgt in self._adjacency[src]:
            # 已存在 → 增强
            existing = self._adjacency[src][tgt]
            existing.weight = min(1.0, existing.weight + weight)
            self._adjacency[tgt][src] = existing
        else:
            # 新建
            edge = Edge(source=src, target=tgt, weight=min(1.0, weight),
                        edge_type=edge_type)
            self._adjacency[src][tgt] = edge
            self._adjacency[tgt][src] = edge

        # 更新边计数
        self.nodes[source].edge_count = len(self._adjacency[source])
        self.nodes[target].edge_count = len(self._adjacency[target])

    def get_edge(self, node_a: str, node_b: str) -> Edge:
        """获取两个节点之间的边"""
        if node_a in self._adjacency:
            return self._adjacency[node_a].get(node_b)
        return None

    def get_neighbors(self, node_id: str) -> Dict[str, Edge]:
        """获取节点的所有邻居 → {neighbor_id: Edge}"""
        return self._adjacency.get(node_id, {})

    @property
    def edge_count(self) -> int:
        """总边数（无向，去重）"""
        count = 0
        seen = set()
        for src, neighbors in self._adjacency.items():
            for tgt in neighbors:
                key = tuple(sorted([src, tgt]))
                if key not in seen:
                    seen.add(key)
                    count += 1
        return count

    def get_all_edges(self) -> List[Tuple[str, str, Edge]]:
        """获取所有边（去重）"""
        edges = []
        seen = set()
        for src, neighbors in self._adjacency.items():
            for tgt, edge in neighbors.items():
                key = tuple(sorted([src, tgt]))
                if key not in seen:
                    seen.add(key)
                    edges.append((src, tgt, edge))
        return edges

    def get_nodes_sorted(self, key_func, reverse: bool = True) -> List[Node]:
        """按指定属性排序返回所有节点"""
        return sorted(self.nodes.values(), key=key_func, reverse=reverse)
