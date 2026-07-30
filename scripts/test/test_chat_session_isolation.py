"""验证聊天 demo 按 session 隔离 history 和 memory retrieval。"""

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


def _post_chat(client: TestClient, session_id: str, text: str) -> list[dict]:
    with client.stream(
        "POST",
        "/api/chat",
        json={"session_id": session_id, "text": text},
    ) as response:
        assert response.status_code == 200
        return _read_sse(response)


def main():
    original_chat_stream = llm_mod.chat_stream
    llm_mod.chat_stream = lambda messages, config: iter(["ok"])

    try:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(_settings(tmp))
            client = TestClient(app)
            session_a = client.post(
                "/api/session",
                json={"client_label": "alice"},
            ).json()["session_id"]
            session_b = client.post(
                "/api/session",
                json={"client_label": "bob"},
            ).json()["session_id"]

            a_first = _post_chat(
                client,
                session_a,
                "Alice private memory keyword zeta42",
            )
            a_turn = next(e for e in a_first if e["type"] == "done")["turn_id"]

            app.state.chat.mem.build_index()

            b_events = _post_chat(
                client,
                session_b,
                "zeta42 private memory",
            )
            b_info = next(e for e in b_events if e["type"] == "info")
            assert b_info["memories_used"] == [], b_info

            a_events = _post_chat(
                client,
                session_a,
                "zeta42 private memory",
            )
            a_info = next(e for e in a_events if e["type"] == "info")
            assert a_info["memories_used"], a_info

            history_a = client.get(
                f"/api/history?session_id={session_a}",
            ).json()
            history_b = client.get(
                f"/api/history?session_id={session_b}",
            ).json()
            assert len(history_a) == 4, history_a
            assert len(history_b) == 2, history_b
            assert {h["session_id"] for h in history_a} == {session_a}
            assert {h["session_id"] for h in history_b} == {session_b}

            memories_b = client.get(
                f"/api/memories?limit=10&session_id={session_b}",
            ).json()
            assert all("zeta42" not in m["text"] for m in memories_b), memories_b

            print(
                {
                    "session_a": session_a,
                    "session_b": session_b,
                    "a_first_turn": a_turn,
                    "a_memories_used": len(a_info["memories_used"]),
                    "b_memories_used": len(b_info["memories_used"]),
                }
            )
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
