"""只跑 Zep/Graphiti，与已有 Memento 结果对比"""

from __future__ import annotations
import asyncio, json, os, re, sys, time, warnings
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"

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


def require_env(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"缺少环境变量: {name}")
    return v


def parse_chat_data(path, limit=None):
    content = path.read_text(encoding="utf-8-sig")
    pairs = []
    for part in re.split(r"【用户提问】", content):
        part = part.strip()
        if not part or "【AI 回答】" not in part:
            continue
        q, a = part.split("【AI 回答】", 1)
        pairs.append({"question": q.strip(), "answer": a.strip()})
        if limit and len(pairs) >= limit:
            break
    return pairs


def memory_text(pair):
    return f"用户问: {pair['question']}\n回答: {pair['answer']}"


def truncate(text, limit=140):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "..." if len(text) > limit else text


def main():
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))

    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"数据: {len(pairs)} 条 Q&A", flush=True)

    # ── 初始化 Graphiti ──
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

    sf_key = require_env("SILICONFLOW_API_KEY")
    sf_url = require_env("SILICONFLOW_API_BASE")
    oc_key = require_env("OPENCODE_API_KEY")
    oc_url = require_env("OPENCODE_API_BASE")

    driver = KuzuDriver(db=":memory:")
    driver._database = "default"  # fix missing attr

    # LLM: OpenCode deepseek-v4-flash（已验证可跑完 146 条）
    llm = OpenAIGenericClient(
        config=LLMConfig(api_key=oc_key, model="deepseek-v4-flash", base_url=oc_url),
        structured_output_mode="json_object",
        max_tokens=16384,
    )

    # ── monkey-patch: DeepSeek json_object 模式返回 {"properties": {...}} ──
    _orig_generate = llm._generate_response

    def _unwrap_schema(obj):
        if not isinstance(obj, dict):
            return obj
        if "properties" in obj and len(obj) <= 2:
            inner = obj["properties"]
            if isinstance(inner, dict):
                sample = next(iter(inner.values()), None)
                if sample is not None and not (isinstance(sample, dict) and "type" in sample and "description" in sample):
                    return inner
        if "$defs" in obj or "definitions" in obj:
            props = obj.get("properties", {})
            if isinstance(props, dict):
                result = {}
                for k, v in props.items():
                    if isinstance(v, dict) and v.get("type") == "array":
                        result[k] = []
                    elif isinstance(v, dict) and v.get("type") == "string":
                        result[k] = ""
                    else:
                        result[k] = v
                return result if result else obj
        return obj

    async def _patched_generate(messages, response_model=None, max_tokens=None, model_size=None, **kwargs):
        result = await _orig_generate(messages, response_model=response_model,
                                       max_tokens=max_tokens, model_size=model_size, **kwargs)
        return _unwrap_schema(result)

    llm._generate_response = _patched_generate
    # Embedding: SiliconFlow Qwen3-4B
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=sf_key, embedding_model="Qwen/Qwen3-Embedding-4B",
        base_url=sf_url, embedding_dim=2560,
    ))
    # Cross-encoder: SiliconFlow BGE reranker
    cross_encoder = OpenAIRerankerClient(config=LLMConfig(
        api_key=sf_key, model="BAAI/bge-reranker-v2-m3", base_url=sf_url,
    ))

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=cross_encoder,
        store_raw_episode_content=True,
        max_coroutines=15,  # OpenCode 限制松
    )

    # ── Build ──
    print("\n=== Zep (Graphiti) ===", flush=True)
    t0 = time.time()

    async def build():
        # 旧版 KuzuDriver.build_indices_and_constraints() 是空操作
        # 需要手动创建 schema + FTS 索引
        print("  创建 Kuzu schema...", flush=True)
        driver.setup_schema()

        print("  创建 FTS 索引...", flush=True)
        conn = __import__("kuzu").Connection(driver.db)
        fts_queries = [
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]
        for q in fts_queries:
            try:
                conn.execute(q)
                print(f"    OK: {q[:60]}...", flush=True)
            except Exception as e:
                print(f"    SKIP: {e}", flush=True)
        conn.close()

        from datetime import datetime, timezone
        from graphiti_core.utils.bulk_utils import RawEpisode
        from graphiti_core.nodes import EpisodeType

        episodes = []
        for i, pair in enumerate(pairs):
            episodes.append(RawEpisode(
                name=f"qa_{i:04d}",
                content=memory_text(pair),
                source_description="benchmark",
                source=EpisodeType.message,
                reference_time=datetime.now(timezone.utc),
            ))

        print(f"  批量写入 {len(episodes)} 条 episode (需要 LLM 实体抽取)...", flush=True)
        # 分批写入，每批 20 条 + 2秒延迟
        batch_size = 20
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]
            print(f"    批次 {i//batch_size + 1}/{(len(episodes) + batch_size - 1)//batch_size} "
                  f"({len(batch)} 条)...", flush=True)
            await graphiti.add_episode_bulk(batch)
            await asyncio.sleep(2)
        print("  写入完成!", flush=True)

    asyncio.run(build())
    build_sec = time.time() - t0
    print(f"  构建耗时: {build_sec:.1f}s", flush=True)

    # ── Query ──
    print(f"\n{'='*60}", flush=True)
    total_ms = 0.0
    for query in QUERIES:
        t0 = time.time()

        async def do_search(q=query):
            return await graphiti.search(query=q, num_results=5)

        results = asyncio.run(do_search())
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed

        print(f"\nQ: {query} [{elapsed:.0f}ms]", flush=True)
        for rank, edge in enumerate(results, 1):
            name = getattr(edge, "name", "")
            fact = getattr(edge, "fact", "")
            score = getattr(edge, "score", None)
            s = f"{score:.4f}" if score is not None else "-"
            # Graphiti 返回的是 EntityEdge，包含 name (关系名) 和 fact (事实描述)
            display = f"{name}: {fact}" if fact else name
            print(f"  {rank}. [{s}] {truncate(display, 120)}", flush=True)

    avg_ms = total_ms / len(QUERIES)
    print(f"\n平均查询: {avg_ms:.0f}ms", flush=True)
    print(f"构建总耗时: {build_sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
