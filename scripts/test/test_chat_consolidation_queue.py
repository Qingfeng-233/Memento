"""验证聊天 demo 的巩固以串行 job 形式执行。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from apps.chat.app import ChatSettings, create_app
from apps.chat.llm import LLMConfig


def _settings(store: str) -> ChatSettings:
    settings = ChatSettings()
    settings.store = store
    settings.embedding_model = "tfidf-svd"
    settings.llm = LLMConfig(mock=True)
    settings.retrieve_k = 5
    settings.retrieve_seed_k = 15
    settings.min_index_size = 1
    settings.idle_minutes = 60.0
    settings.idle_check_interval = 3600.0
    settings.enable_concept_graph = False
    settings.history_turns = 6
    return settings


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_settings(tmp))
        client = TestClient(app)

        app.state.chat.mem.add_node(
            "consolidation queue memory",
            tags=["session:test"],
            source="chat",
        )
        assert len(app.state.chat.mem._pending_nodes) == 1

        result = client.post("/api/consolidate").json()
        assert result["ok"] is True, result
        job = result["job"]
        assert job["job_id"].startswith("job_")
        assert job["trigger"] == "manual"
        assert job["status"] == "done"
        assert any("build_index" in step for step in job["steps"])
        assert len(app.state.chat.mem._pending_nodes) == 0
        assert app.state.chat.mem.graph.node_count == 1

        log = client.get("/api/consolidation-log").json()
        assert log[-1]["job_id"] == job["job_id"]
        assert log[-1]["status"] == "done"

        print(
            {
                "job_id": job["job_id"],
                "steps": job["steps"],
                "log_count": len(log),
            }
        )


if __name__ == "__main__":
    main()
