"""验证 KeyAtten 能否把事件文本压缩成可作为记忆节点的短语。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memento.index.keyatten_extractor import MemoryKeywordExtractor


MEMORY_TEXT = "上周客户要求我们下周完成项目交付，但预算审批尚未通过，团队正在调整实施方案。"
QUERY_TEXT = "客户在问项目进度，预算审批现在有结果了吗？"
ANCHORS = ("客户", "下周", "项目", "交付", "预算", "审批", "实施方案")


def matched_anchors(keywords: list[str]) -> list[str]:
    return [anchor for anchor in ANCHORS if any(anchor in keyword for keyword in keywords)]


def main() -> None:
    extractor = MemoryKeywordExtractor(
        model_path=str(ROOT / "models" / "Qwen3-Embedding-0.6B"),
        device="cuda",
        dtype="float16",
        default_top_k=10,
        phrase_merge_enabled=True,
        cache_enabled=True,
        cache_dir=ROOT / "data" / "keyatten_cache",
    )

    memory_nodes = extractor.extract(MEMORY_TEXT)
    query_nodes = extractor.extract(QUERY_TEXT)
    shared_nodes = sorted(set(memory_nodes) & set(query_nodes))

    print("=== 事件记忆原文 ===")
    print(MEMORY_TEXT)
    print(f"KeyAtten 节点: {memory_nodes}")
    print(f"预期锚点命中: {matched_anchors(memory_nodes)}")

    print("\n=== 当前查询原文 ===")
    print(QUERY_TEXT)
    print(f"KeyAtten 节点: {query_nodes}")
    print(f"预期锚点命中: {matched_anchors(query_nodes)}")

    print("\n=== 两段文本的直接重合节点 ===")
    print(shared_nodes or "（无完全相同短语；后续需由概念相似度或模式关联连接）")


if __name__ == "__main__":
    main()
