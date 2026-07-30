"""验证 observability API 聚合 turn / memory / job / retrieval 状态。"""

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
                json={"client_label": "observability-test"},
            ).json()["session_id"]

            with client.stream(
                "POST",
                "/api/chat",
                json={"session_id": session_id, "text": "observability memory"},
            ) as response:
                assert response.status_code == 200
                events = _read_sse(response)
            done = next(e for e in events if e["type"] == "done")

            job = client.post("/api/consolidate").json()["job"]
            obs = client.get(f"/api/observability?session_id={session_id}").json()

            assert obs["session_id"] == session_id
            assert obs["turns"], obs
            assert obs["turns"][0]["turn_id"] == done["turn_id"]
            assert any(m["node_id"] == done["node_id"] for m in obs["memory_events"])
            assert any(j["job_id"] == job["job_id"] for j in obs["jobs"])
            assert any(r["turn_id"] == done["turn_id"] for r in obs["retrievals"])
            assert obs["errors"] == []

            print(
                {
                    "turns": len(obs["turns"]),
                    "memory_events": len(obs["memory_events"]),
                    "jobs": len(obs["jobs"]),
                    "retrievals": len(obs["retrievals"]),
                }
            )
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
