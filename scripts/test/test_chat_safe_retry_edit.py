"""验证 retry/edit 按 turn 精确 supersede，而不是物理删除旧记忆。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from apps.chat import llm as llm_mod
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


def _read_sse(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _post_chat(client: TestClient, payload: dict) -> list[dict]:
    with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        return _read_sse(response)


def main():
    original_chat_stream = llm_mod.chat_stream
    llm_mod.chat_stream = lambda messages, config: iter(["ok"])

    try:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(_settings(tmp))
            client = TestClient(app)
            session_id = client.post(
                "/api/session",
                json={"client_label": "safe-retry-test"},
            ).json()["session_id"]

            first_events = _post_chat(
                client,
                {"session_id": session_id, "text": "旧偏好 zeta-retry"},
            )
            first_done = next(e for e in first_events if e["type"] == "done")

            retry_events = _post_chat(
                client,
                {
                    "session_id": session_id,
                    "text": "新偏好 zeta-retry",
                    "replace_last": True,
                    "edit_of_turn_id": first_done["turn_id"],
                },
            )
            retry_done = next(e for e in retry_events if e["type"] == "done")

            visible_history = client.get(
                f"/api/history?session_id={session_id}",
            ).json()
            assert len(visible_history) == 2, visible_history
            assert visible_history[0]["turn_id"] == retry_done["turn_id"]

            all_history = app.state.chat.history
            old_entries = [
                h for h in all_history if h.get("turn_id") == first_done["turn_id"]
            ]
            assert len(old_entries) == 2, old_entries
            assert {h["status"] for h in old_entries} == {"superseded"}
            assert all(
                h["superseded_by_turn_id"] == retry_done["turn_id"]
                for h in old_entries
            )

            app.state.chat.mem.build_index()
            old_node = app.state.chat.mem.graph.get_node(first_done["node_id"])
            new_node = app.state.chat.mem.graph.get_node(retry_done["node_id"])
            assert old_node.status == "superseded"
            assert old_node.superseded_by == retry_done["node_id"]
            assert new_node.status == "active"

            results = app.state.chat.mem.query("zeta-retry", k=5)
            result_ids = [r["id"] for r in results]
            assert first_done["node_id"] not in result_ids
            assert retry_done["node_id"] in result_ids

            print(
                {
                    "old_turn": first_done["turn_id"],
                    "new_turn": retry_done["turn_id"],
                    "old_node": first_done["node_id"],
                    "new_node": retry_done["node_id"],
                    "query_ids": result_ids,
                }
            )
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
