"""
数据模型定义 - 节点 (Node) 和边 (Edge)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Node:
    """
    记忆节点 - 代表一个"状态片段"或"概念"

    Attributes:
        id: 唯一标识
        text: 自然语言描述
        vector: 高维嵌入向量（由嵌入模型生成）
        importance (ω): 重要性系数 ∈ [0, 1]，长期固有价值
        vitality (λ): 生命力 ∈ [0, 1]，瞬时活跃度
        access_count: 被检索/激活的累计次数
        edge_count: 连接的边数
        tags: 标签列表
        created_at: 创建时间
        source: 来源 (import / chat / manual)
        status: 状态 (active / dormant / cold / superseded)
        superseded_by: 被取代指向（融合后源节点指向融合节点）
        fused_from: 融合血缘（仅融合节点非空，记录源节点 id 列表）
    """
    id: str
    text: str
    vector: Optional[np.ndarray] = None
    importance: float = 0.5        # ω
    vitality: float = 1.0          # λ
    access_count: int = 0
    edge_count: int = 0
    tags: list = field(default_factory=list)
    created_at: Optional[str] = None
    source: str = "import"
    status: str = "active"
    superseded_by: Optional[str] = None
    fused_from: list = field(default_factory=list)


@dataclass
class Edge:
    """
    边 - 节点间的经验连接

    Attributes:
        source: 源节点 ID
        target: 目标节点 ID
        weight (w): 边强度 ∈ [0, 1]
        edge_type: 边类型 (cooccurrence / manual / explore)
    """
    source: str
    target: str
    weight: float = 0.1
    edge_type: str = "cooccurrence"
