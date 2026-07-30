"""验证聊天 demo 的 turn_id / node_id 回合绑定。"""

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
    settings.min_index_size = 3
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
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    return events


def _post_chat(client: TestClient, payload: dict) -> list[dict]:
    with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        return _read_sse(response)


def main():
    original_chat_stream = llm_mod.chat_stream

    def fake_chat_stream(messages, config):
        for message in messages:
            assert set(message) == {"role", "content"}, message
        return iter(["你好，", "已记录。"])

    llm_mod.chat_stream = fake_chat_stream

    try:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(_settings(tmp))
            client = TestClient(app)
            session_id = client.post(
                "/api/session",
                json={"client_label": "turn-model-test"},
            ).json()["session_id"]

            first_events = _post_chat(
                client,
                {"session_id": session_id, "text": "第一轮"},
            )
            first_info = next(e for e in first_events if e["type"] == "info")
            first_done = next(e for e in first_events if e["type"] == "done")
            first_content = [e for e in first_events if e["type"] == "content"]

            first_turn_id = first_done["turn_id"]
            first_node_id = first_done["node_id"]
            assert first_turn_id.startswith("turn_")
            assert first_info["turn_id"] == first_turn_id
            assert all(e["turn_id"] == first_turn_id for e in first_content)
            assert first_node_id

            retry_events = _post_chat(
                client,
                {
                    "session_id": session_id,
                    "text": "第一轮修订",
                    "replace_last": True,
                    "retry_of_turn_id": first_turn_id,
                },
            )
            retry_done = next(e for e in retry_events if e["type"] == "done")
            retry_turn_id = retry_done["turn_id"]
            retry_node_id = retry_done["node_id"]

            history = client.get(f"/api/history?session_id={session_id}").json()
            assert len(history) == 2, history
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "assistant"
            assert history[0]["turn_id"] == retry_turn_id
            assert history[1]["turn_id"] == retry_turn_id
            assert history[0]["session_id"] == session_id
            assert history[1]["session_id"] == session_id
            assert history[0]["supersedes_turn_id"] == first_turn_id
            assert history[1]["node_id"] == retry_node_id

            pending_by_id = {
                p["id"]: p for p in app.state.chat.mem._pending_nodes
            }
            assert pending_by_id[first_node_id]["status"] == "superseded"
            assert pending_by_id[first_node_id]["superseded_by"] == retry_node_id
            assert pending_by_id[retry_node_id].get("status", "active") == "active"

            print(
                {
                    "first_turn_id": first_turn_id,
                    "retry_turn_id": retry_turn_id,
                    "retry_node_id": retry_node_id,
                    "pending_ids": list(pending_by_id),
                }
            )
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
