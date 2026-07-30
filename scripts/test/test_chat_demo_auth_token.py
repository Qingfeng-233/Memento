"""验证聊天 demo 的可选 token 保护。"""

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


TOKEN = "test-token"


def _settings(store: str) -> ChatSettings:
    settings = ChatSettings()
    settings.store = store
    settings.embedding_model = "tfidf-svd"
    settings.auth_token = TOKEN
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
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def main():
    original_chat_stream = llm_mod.chat_stream
    llm_mod.chat_stream = lambda messages, config: iter(["ok"])

    try:
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(_settings(tmp)))

            config = client.get("/api/config").json()
            assert config["auth_required"] is True

            session_id = client.post(
                "/api/session",
                json={"client_label": "auth-test"},
            ).json()["session_id"]

            assert client.get(f"/api/history?session_id={session_id}").status_code == 401
            assert (
                client.get(
                    f"/api/history?session_id={session_id}",
                    headers={"X-Memento-Token": "wrong"},
                ).status_code
                == 401
            )

            with client.stream(
                "POST",
                "/api/chat",
                json={"session_id": session_id, "text": "token protected chat"},
            ) as response:
                assert response.status_code == 401

            with client.stream(
                "POST",
                "/api/chat",
                headers={"X-Memento-Token": TOKEN},
                json={"session_id": session_id, "text": "token protected chat"},
            ) as response:
                assert response.status_code == 200
                events = _read_sse(response)
            assert next(e for e in events if e["type"] == "done")["node_id"]

            history = client.get(
                f"/api/history?session_id={session_id}",
                headers={"X-Memento-Token": TOKEN},
            ).json()
            assert len(history) == 2, history

            print({"session_id": session_id, "history_len": len(history)})
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
