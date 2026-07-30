"""
验证 Letta 使用 OpenCode LLM + SiliconFlow Qwen3 embedding 的记忆写入/搜索。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from letta_client import Letta
from letta_client.types.embedding_config import EmbeddingConfig
from letta_client.types.llm_config import LlmConfig


ROOT = Path(__file__).resolve().parent.parent.parent
LETTA_BASE_URL = "http://localhost:8283"
PROVIDER_NAME = "opencode-deepseek-v4-flash"
PROBE_TAG = "memento_compare_probe"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def ensure_opencode_provider(client: Letta):
    providers = client.providers.list(name=PROVIDER_NAME)
    if providers:
        provider = providers[0]
        return client.providers.modify(
            provider.id,
            api_key=require_env("OPENCODE_API_KEY"),
            base_url=require_env("OPENCODE_API_BASE"),
        )

    return client.providers.create(
        name=PROVIDER_NAME,
        provider_type="openai",
        api_key=require_env("OPENCODE_API_KEY"),
        base_url=require_env("OPENCODE_API_BASE"),
    )


def create_probe_agent(client: Letta):
    return client.agents.create(
        name=f"memento_compare_fixed_{int(time.time())}",
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
    client = Letta(base_url=LETTA_BASE_URL)

    provider = ensure_opencode_provider(client)
    print(f"provider: {provider.id}")

    agent = create_probe_agent(client)
    print(f"agent: {agent.id}")

    passages = client.agents.passages.create(
        agent.id,
        text=(
            "测试记忆: 钢琴连接电脑需要 MIDI 转 USB 线，"
            "Korg D1 没有 USB 口时要用圆头 MIDI 线。"
        ),
        tags=[PROBE_TAG],
    )
    print(f"passage: {passages[0].id if passages else 'none'}")

    result = client.agents.passages.search(
        agent.id,
        query="钢琴连电脑需要什么线",
        tags=PROBE_TAG,
        top_k=3,
    )
    print(f"hits: {result.count}")
    for hit in result.results:
        print(f"- {hit.content}")


if __name__ == "__main__":
    main()
