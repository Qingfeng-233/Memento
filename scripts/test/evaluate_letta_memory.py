"""
Letta 记忆命中评测。

与 compare_systems.py 使用同一数据集和查询集，但只评测 Letta。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from letta_client import Letta
from letta_client.types.embedding_config import EmbeddingConfig
from letta_client.types.llm_config import LlmConfig

from verify_letta_memory import PROVIDER_NAME, ensure_opencode_provider, require_env


ROOT = Path(__file__).resolve().parent.parent.parent
LETTA_BASE_URL = "http://localhost:8283"
OUTPUT = ROOT / "scripts" / "test" / "letta_memory_eval_output.txt"

QUERIES = [
    "容器启动时配置丢失怎么排查",
    "钢琴连电脑需要什么线和软件",
    "怎么提高学习效率防止晚上崩盘",
    "为什么流行歌都是情情爱爱",
    "手机传文件到电脑用什么软件",
    "独立游戏开发者要不要学美术",
    "梯子和局域网冲突怎么解决",
    "怎么有效休息不会浪费意志力",
]


def parse_chat_data(filepath: Path) -> list[dict[str, str]]:
    content = filepath.read_text(encoding="utf-8-sig")
    parts = re.split(r"【用户提问】", content)
    pairs = []
    for part in parts:
        part = part.strip()
        if not part or "【AI 回答】" not in part:
            continue
        question, answer = part.split("【AI 回答】", 1)
        question, answer = question.strip(), answer.strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs


def truncate(text: str, limit: int = 100) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "..." if len(text) > limit else text


def create_eval_agent(client: Letta):
    return client.agents.create(
        name=f"memento_compare_letta_eval_{int(time.time())}",
        llm_config=LlmConfig(
            model="deepseek-v4-flash",
            model_endpoint_type="openai",
            model_endpoint=require_env("OPENCODE_API_BASE"),
            provider_name=PROVIDER_NAME,
            provider_category="byok",
            context_window=30000,
            temperature=0.0,
            max_tokens=1000,
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint=require_env("SILICONFLOW_API_BASE"),
            embedding_model="Qwen/Qwen3-Embedding-4B",
            embedding_dim=2560,
            embedding_chunk_size=300,
        ),
        include_base_tools=False,
        include_default_source=False,
        memory_blocks=[],
    )


def main() -> None:
    load_dotenv(ROOT / ".env")

    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    client = Letta(base_url=LETTA_BASE_URL)
    provider = ensure_opencode_provider(client)
    agent = create_eval_agent(client)

    lines = [
        "# Letta 记忆命中评测结果",
        f"# 日期: {time.strftime('%Y-%m-%d %H:%M')}",
        f"# 数据: {len(pairs)} Q&A pairs",
        "# LLM: OpenCode deepseek-v4-flash via BYOK provider",
        "# Embedding: SiliconFlow Qwen/Qwen3-Embedding-4B",
        f"# Provider: {provider.id}",
        f"# Agent: {agent.id}",
        "",
    ]

    print(f"Letta agent: {agent.id}", flush=True)
    print(f"写入 {len(pairs)} 条记忆...", flush=True)

    t0 = time.time()
    for index, pair in enumerate(pairs):
        text = f"用户问: {pair['question']}\n回答: {pair['answer']}"
        client.agents.passages.create(
            agent.id,
            text=text,
            tags=["memento_compare_eval", f"idx:{index}"],
        )
        if (index + 1) % 20 == 0 or index == len(pairs) - 1:
            print(f"  写入进度: {index + 1}/{len(pairs)}", flush=True)

    build_seconds = time.time() - t0
    lines.append(f"BuildTime: {build_seconds:.1f}s")
    lines.append("")
    print(f"写入完成: {build_seconds:.1f}s", flush=True)

    for query in QUERIES:
        t0 = time.time()
        result = client.agents.passages.search(
            agent.id,
            query=query,
            tags="memento_compare_eval",
            top_k=5,
        )
        query_ms = (time.time() - t0) * 1000
        lines.append(f"Q: {query}")
        lines.append(f"  Time: {query_ms:.0f}ms")
        print(f"\nQ: {query} [{query_ms:.0f}ms]", flush=True)
        for rank, hit in enumerate(result.results, 1):
            content = truncate(hit.content)
            lines.append(f"  {rank}. {content}")
            print(f"  {rank}. {content}", flush=True)
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已保存到: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
