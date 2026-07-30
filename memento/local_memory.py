"""本地 Q/A 记忆服务的 SQLite 持久化与检索编排。"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from memento.api import Memento


@dataclass(frozen=True)
class MemoryRecord:
    """一条可持久化的问答记忆。"""

    id: str
    question: str
    answer: str
    created_at: str


class SQLiteMemoryRepository:
    """以单个 SQLite 文件保存问答记忆。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )

    def add(self, question: str, answer: str) -> MemoryRecord:
        record = MemoryRecord(
            id=uuid.uuid4().hex,
            question=question,
            answer=answer,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO memories (id, question, answer, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (record.id, record.question, record.answer, record.created_at),
                )
        return record

    def list_all(self) -> list[MemoryRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, question, answer, created_at
                FROM memories
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [MemoryRecord(**dict(row)) for row in rows]


class LocalMemoryService:
    """将 SQLite 中的 Q/A 重建为 Memento 索引的本地服务内核。"""

    def __init__(self, database_path: str | Path) -> None:
        self.repository = SQLiteMemoryRepository(database_path)
        self._lock = threading.RLock()
        self._memory = Memento(embedding_model="tfidf-svd")
        self._records: dict[str, MemoryRecord] = {}
        self._rebuild_index()

    @staticmethod
    def _node_text(record: MemoryRecord) -> str:
        return f"用户问: {record.question}\n回答: {record.answer}"

    def _rebuild_index(self) -> None:
        records = self.repository.list_all()
        memory = Memento(embedding_model="tfidf-svd")
        for record in records:
            memory.add_node(
                text=self._node_text(record),
                node_id=record.id,
                source="local-service",
                created_at=record.created_at,
            )
        if records:
            memory.build_index()
            memory.build_concept_graph(keyword_method="statistical")
            memory.build_event_edges_from_keywords()
        self._memory = memory
        self._records = {record.id: record for record in records}

    @property
    def memory_count(self) -> int:
        with self._lock:
            return len(self._records)

    def add(self, question: str, answer: str) -> MemoryRecord:
        with self._lock:
            record = self.repository.add(question, answer)
            self._rebuild_index()
            return record

    def search(self, query: str, limit: int) -> list[dict]:
        with self._lock:
            if not self._records:
                return []
            hits = self._memory.query_associative(
                query,
                k=limit,
                seed_k=min(limit, 5),
            )
            return [
                {
                    "id": record.id,
                    "question": record.question,
                    "answer": record.answer,
                    "score": hit["score"],
                    "rag_score": hit["rag_score"],
                    "event_score": hit["event_score"],
                    "concept_score": hit["concept_score"],
                }
                for hit in hits
                if (record := self._records.get(hit["id"])) is not None
            ]
