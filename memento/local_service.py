"""面向应用调用方的最小本地 Q/A 记忆 HTTP 服务。"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from memento.local_memory import LocalMemoryService


class AddMemoryRequest(BaseModel):
    question: str = Field(max_length=20_000)
    answer: str = Field(max_length=20_000)

    @field_validator("question", "answer")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class SearchRequest(BaseModel):
    query: str = Field(max_length=20_000)
    limit: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def require_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


def create_app(database_path: str | Path = "data/memento.db") -> FastAPI:
    """创建只包含写入、查询和健康检查的本地服务。"""
    memory_service = LocalMemoryService(database_path)
    app = FastAPI(
        title="Memento Local Memory Service",
        description="本地 Q/A 写入与语义查询服务",
        version="0.1.0",
    )
    app.state.memory_service = memory_service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "memory_count": memory_service.memory_count}

    @app.post("/memories")
    def add_memory(request: AddMemoryRequest) -> dict:
        record = memory_service.add(request.question, request.answer)
        return {"id": record.id, "created_at": record.created_at}

    @app.post("/search")
    def search(request: SearchRequest) -> dict:
        return {"results": memory_service.search(request.query, request.limit)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m memento.local_service",
        description="启动 Memento 本地 Q/A 记忆服务",
    )
    parser.add_argument("--db", default="data/memento.db", help="SQLite 数据库文件路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.db), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
