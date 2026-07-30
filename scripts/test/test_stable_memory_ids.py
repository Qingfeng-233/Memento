"""验证默认 node_id 在删除后不会复用已有编号。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from memento.api import Memento


def main():
    mem = Memento()

    texts = [
        "苹果 香蕉 记忆",
        "钢琴 电脑 记忆",
        "小说 魔法 记忆",
        "健身 睡眠 记忆",
    ]
    ids = [mem.add_node(text) for text in texts]
    assert ids == ["mem_000000", "mem_000001", "mem_000002", "mem_000003"], ids
    mem.build_index()

    mem.graph.nodes.pop("mem_000001", None)
    mem.vector_index._id_map[1] = ""

    new_id = mem.add_node("删除中间节点后新增")
    assert new_id == "mem_000004", new_id
    assert len(mem._pending_nodes) == 1
    assert mem._pending_nodes[0]["id"] == "mem_000004"

    live_id = None
    if mem._index_built:
        mem.build_index()
        live_id = mem.add_node_live("实时新增节点")
        assert live_id == "mem_000005", live_id

    print(
        {
            "initial_ids": ids,
            "new_pending_id": new_id,
            "live_id": live_id,
            "next_node_seq": mem._next_node_seq,
        }
    )


if __name__ == "__main__":
    main()
