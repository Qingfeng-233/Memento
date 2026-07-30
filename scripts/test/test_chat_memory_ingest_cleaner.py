"""验证聊天入库前会清理 think / mock / stream error 文本。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from apps.chat import ingest as ingest_mod
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
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def main():
    cleaned = ingest_mod.clean_reply_for_memory(
        "可见回答\n<think>不该入库</think>\n[Stream Error: timeout]\n结束"
    )
    assert "不该入库" not in cleaned
    assert "Stream Error" not in cleaned
    assert "可见回答" in cleaned
    assert "结束" in cleaned

    original_chat_stream = llm_mod.chat_stream
    llm_mod.chat_stream = lambda messages, config: iter(
        [
            "[mock 流式回复] 调试信息\n",
            "这是未接真实大模型的 Mock 回复。\n",
            "<think>mock 思考链</think>\n",
            "不应写入的 mock 正文",
        ]
    )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(_settings(tmp))
            client = TestClient(app)
            session_id = client.post(
                "/api/session",
                json={"client_label": "cleaner-test"},
            ).json()["session_id"]

            with client.stream(
                "POST",
                "/api/chat",
                json={"session_id": session_id, "text": "记住我的偏好是低噪音"},
            ) as response:
                assert response.status_code == 200
                _read_sse(response)

            pending = app.state.chat.mem._pending_nodes
            assert len(pending) == 1, pending
            text = pending[0]["text"]
            assert text == "用户问: 记住我的偏好是低噪音", text
            assert "think" not in text.lower()
            assert "Mock" not in text
            assert "未接真实大模型" not in text

            print({"pending_text": text, "tags": pending[0]["tags"]})
    finally:
        llm_mod.chat_stream = original_chat_stream


if __name__ == "__main__":
    main()
