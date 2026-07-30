"""验证 ChatStore 收口 session / history / consolidation job 持久化。"""

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


def main():
    original_chat_stream = llm_mod.chat_stream
    llm_mod.chat_stream = lambda messages, config: iter(["ok"])

    try:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(_settings(tmp))
            client = TestClient(app)
            session_id = client.post(
                "/api/session",
                json={"client_label": "store-test"},
            ).json()["session_id"]

            with client.stream(
                "POST",
                "/api/chat",
                json={"session_id": session_id, "text": "store abstraction"},
            ) as response:
                assert response.status_code == 200
                _read_sse(response)

            job = client.post("/api/consolidate").json()["job"]

            reloaded_app = create_app(_settings(tmp))
            reloaded = reloaded_app.state.chat
            assert session_id in reloaded.sessions
            assert len(reloaded.history) == 2, reloaded.history
            assert reloaded.history[0]["session_id"] == session_id
            assert reloaded.consolidation_log[-1]["job_id"] == job["job_id"]

            store = reloaded.chat_store
            assert store.history_path.exists()
            assert store.sessions_path.exists()
            assert store.jobs_path.exists()

            print(
                {
                    "session_id": session_id,
                    "history_len": len(reloaded.history),
                    "last_job": job["job_id"],
                }
            )
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
