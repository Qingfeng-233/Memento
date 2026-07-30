"""验证本地 Q/A 服务的持久化、完整图联想与请求边界。"""

import tempfile
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memento.local_service import create_app


def test_local_service() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "memento.db"
        client = TestClient(create_app(database_path))

        assert client.get("/health").json() == {"status": "ok", "memory_count": 0}
        assert client.post("/search", json={"query": "如何备份？"}).json() == {
            "results": []
        }

        backup = client.post(
            "/memories",
            json={
                "question": "如何备份项目？",
                "answer": "使用版本控制并保留异地备份。",
            },
        )
        music = client.post(
            "/memories",
            json={
                "question": "学习时怎样专注？",
                "answer": "关闭通知，按固定时段完成一项任务。",
            },
        )
        history = client.post(
            "/memories",
            json={
                "question": "版本控制除了保存代码还能做什么？",
                "answer": "版本控制可以追踪修改历史，并帮助恢复旧版本。",
            },
        )
        assert backup.status_code == 200
        assert music.status_code == 200
        assert history.status_code == 200
        assert database_path.exists()
        assert client.get("/health").json()["memory_count"] == 3

        service = client.app.state.memory_service
        memory = service._memory
        backup_id = backup.json()["id"]
        history_id = history.json()["id"]
        event_edge = memory.graph.get_edge(backup_id, history_id)
        assert event_edge is not None
        assert event_edge.edge_type == "keyword"
        assert memory.concept_graph.concepts
        assert memory.concept_graph.event_to_concepts
        assert any(memory.concept_graph.concept_edges.values())

        associative = memory.query_associative(
            "代码历史如何恢复",
            k=3,
            seed_k=1,
        )
        associated_backup = next(hit for hit in associative if hit["id"] == backup_id)
        assert associated_backup["event_score"] > 0
        assert associated_backup["concept_score"] > 0

        results = client.post(
            "/search", json={"query": "项目代码怎么备份", "limit": 1}
        )
        assert results.status_code == 200
        assert results.json()["results"][0]["id"] == backup.json()["id"]
        assert {
            "rag_score",
            "event_score",
            "concept_score",
        } <= results.json()["results"][0].keys()

        restarted = TestClient(create_app(database_path))
        after_restart = restarted.post(
            "/search", json={"query": "异地备份项目怎么做", "limit": 3}
        )
        assert any(
            item["answer"] == "使用版本控制并保留异地备份。"
            for item in after_restart.json()["results"]
        )
        restarted_memory = restarted.app.state.memory_service._memory
        assert restarted_memory.graph.get_edge(backup_id, history_id) is not None
        assert restarted_memory.concept_graph.concepts

        assert client.post(
            "/memories", json={"question": " ", "answer": "有效回答"}
        ).status_code == 422
        assert client.post(
            "/memories", json={"question": "有效问题", "answer": " "}
        ).status_code == 422
        assert client.post("/search", json={"query": " ", "limit": 5}).status_code == 422
        assert client.post("/search", json={"query": "有效查询", "limit": 0}).status_code == 422


if __name__ == "__main__":
    test_local_service()
    print("local service tests: ok")
