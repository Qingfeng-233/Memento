"""聊天 demo 的持久化抽象。

第一阶段仍使用 JSON 文件，调用方不直接读写具体文件名。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class JsonChatStore:
    """基于单目录 JSON 文件的 ChatStore 实现。"""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.history_path = self.directory / "chat_history.json"
        self.sessions_path = self.directory / "chat_sessions.json"
        self.jobs_path = self.directory / "chat_consolidation_jobs.json"

    def _read_json(self, path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json(self, path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_sessions(self) -> dict[str, dict]:
        data = self._read_json(self.sessions_path, [])
        return {
            item["session_id"]: item
            for item in data
            if isinstance(item, dict) and item.get("session_id")
        }

    def save_sessions(self, sessions: dict[str, dict]) -> None:
        self._write_json(self.sessions_path, list(sessions.values()))

    def create_session(
        self,
        sessions: dict[str, dict],
        client_label: Optional[str] = None,
    ) -> dict:
        now = _now_text()
        session = {
            "session_id": f"session_{uuid.uuid4().hex}",
            "owner_id": f"owner_{uuid.uuid4().hex}",
            "client_label": client_label or "",
            "created_at": now,
            "updated_at": now,
        }
        sessions[session["session_id"]] = session
        self.save_sessions(sessions)
        return session

    def load_history(self) -> list[dict]:
        data = self._read_json(self.history_path, [])
        return data if isinstance(data, list) else []

    def save_history(self, history: list[dict]) -> None:
        self._write_json(self.history_path, history)

    def load_consolidation_jobs(self) -> list[dict]:
        data = self._read_json(self.jobs_path, [])
        return data if isinstance(data, list) else []

    def append_consolidation_job(self, job: dict, limit: int = 20) -> list[dict]:
        jobs = self.load_consolidation_jobs()
        jobs.append(job)
        if len(jobs) > limit:
            jobs = jobs[-limit:]
        self._write_json(self.jobs_path, jobs)
        return jobs
