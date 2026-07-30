"""mem0 完整 benchmark：146 条 Q&A + 8 个查询"""

import json, os, re, shutil, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

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


def truncate(text, limit=200):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "..." if len(text) > limit else text


def main():
    from mem0 import Memory

    # 清理旧数据
    db_path = str(ROOT / "benchmark" / "_mem0_bench")
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    config = {
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "Qwen/Qwen3-Embedding-4B",
                "api_key": os.environ["SILICONFLOW_API_KEY"],
                "openai_base_url": os.environ["SILICONFLOW_API_BASE"],
                "embedding_dims": 2560,
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "deepseek-v4-flash",
                "api_key": os.environ["OPENCODE_API_KEY"],
                "openai_base_url": os.environ["OPENCODE_API_BASE"],
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_bench",
                "embedding_model_dims": 2560,
                "path": db_path,
            }
        },
        "version": "v1.1",
    }

    print("=" * 60)
    print("  mem0 Benchmark")
    print("  Embedding: SiliconFlow Qwen3-Embedding-4B (2560d)")
    print("  LLM: OpenCode deepseek-v4-flash (事实提取)")
    print("  VectorStore: Qdrant (local)")
    print("=" * 60)

    m = Memory.from_config(config)

    # 加载数据
    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"\n数据: {len(pairs)} 条 Q&A")

    # ── Build: 逐条 add ──
    print(f"\n{'─' * 60}")
    print("  Build: 逐条添加 (LLM 事实提取 + embedding)")
    print(f"{'─' * 60}")

    build_start = time.time()
    add_times = []
    errors = 0
    for i, pair in enumerate(pairs):
        text = f"用户问: {pair['question']}\n回答: {pair['answer']}"
        t0 = time.time()
        try:
            result = m.add(text, user_id="benchmark")
            elapsed = time.time() - t0
            add_times.append(elapsed)

            # 提取事实数量
            facts = result.get("results", [])
            n_facts = sum(1 for f in facts if f.get("event") in ("ADD", "UPDATE"))

            if (i + 1) % 10 == 0 or i == 0:
                avg_t = sum(add_times) / len(add_times)
                eta = avg_t * (len(pairs) - i - 1) / 60
                print(f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s (avg {avg_t:.1f}s, ETA {eta:.0f}min) +{n_facts} facts", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            add_times.append(elapsed)
            errors += 1
            if errors <= 3:
                print(f"  [{i+1}] ERROR ({elapsed:.1f}s): {str(e)[:80]}", flush=True)

    build_sec = time.time() - build_start
    print(f"\nBuild 完成: {build_sec:.1f}s ({build_sec/60:.1f}min)")
    print(f"  平均每条: {sum(add_times)/len(add_times):.1f}s")
    print(f"  错误: {errors}/{len(pairs)}")

    # 查看所有记忆
    all_memories = m.get_all(user_id="benchmark")
    mem_list = all_memories.get("results", [])
    print(f"  总记忆数: {len(mem_list)}")

    # ── Search: 8 个查询 ──
    print(f"\n{'=' * 60}")
    print("  Search: 8 个查询")
    print(f"{'=' * 60}")

    top_k = 5
    total_ms = 0
    for qi, query in enumerate(QUERIES, 1):
        t0 = time.time()
        results = m.search(query, top_k=top_k, filters={"user_id": "benchmark"})
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed

        hits = results.get("results", [])

        print(f"\n{'─' * 70}")
        print(f"  Q{qi}: {query}  [{elapsed:.0f}ms]")
        print(f"{'─' * 70}")

        if not hits:
            print("  (空结果)")
            continue

        for rank, hit in enumerate(hits[:top_k], 1):
            mem = hit.get("memory", "")
            score = hit.get("score", None)
            score_str = f"{score:.4f}" if score is not None else "-"
            print(f"  #{rank}  score={score_str}")
            print(f"      {truncate(mem, 160)}")

    avg_ms = total_ms / len(QUERIES)
    print(f"\n{'═' * 60}")
    print(f"  平均查询延迟: {avg_ms:.0f}ms")
    print(f"{'═' * 60}")

    # 保存结果
    results_path = ROOT / "benchmark" / "results" / "mem0_benchmark.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "system": "mem0",
        "build_sec": round(build_sec, 1),
        "build_avg_per_item": round(sum(add_times) / len(add_times), 1),
        "total_memories": len(mem_list),
        "errors": errors,
        "avg_query_ms": round(avg_ms, 1),
        "queries": {},
    }
    results_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {results_path}")

    # 清理
    shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    main()
