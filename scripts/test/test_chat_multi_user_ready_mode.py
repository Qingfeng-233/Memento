"""验证 multi-user ready 模式的最小 ownership 边界。"""

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
    settings.retrieve_k = 3
    settings.retrieve_seed_k = 10
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
            client = TestClient(create_app(_settings(tmp)))
            session_a = client.post(
                "/api/session",
                json={"client_label": "owner-a"},
            ).json()
            session_b = client.post(
                "/api/session",
                json={"client_label": "owner-b"},
            ).json()

            with client.stream(
                "POST",
                "/api/chat",
                json={
                    "session_id": session_a["session_id"],
                    "text": "owner protected memory",
                },
            ) as response:
                assert response.status_code == 200
                events = _read_sse(response)
            node_id = next(e for e in events if e["type"] == "done")["node_id"]

            memories_b = client.get(
                f"/api/memories?limit=10&session_id={session_b['session_id']}",
            ).json()
            assert all(m["id"] != node_id for m in memories_b), memories_b

            forbidden = client.delete(
                f"/api/node/{node_id}?session_id={session_b['session_id']}",
            )
            assert forbidden.status_code == 403, forbidden.text

            allowed = client.delete(
                f"/api/node/{node_id}?session_id={session_a['session_id']}",
            )
            assert allowed.status_code == 200, allowed.text

            print(
                {
                    "owner_a": session_a["owner_id"],
                    "owner_b": session_b["owner_id"],
                    "node_id": node_id,
                    "forbidden_status": forbidden.status_code,
                }
            )
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
