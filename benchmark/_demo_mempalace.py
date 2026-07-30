"""MemPalace 演示：构建索引 + 8 个查询，展示返回结果和分数

修复版：使用 SiliconFlow Qwen3-Embedding-4B（中文原生）替代默认的 all-MiniLM-L6-v2（英文为主）
"""

import os, re, sys, time, shutil, warnings
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


def memory_text(pair):
    return f"用户问: {pair['question']}\n回答: {pair['answer']}"


def truncate(text, limit=200):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "..." if len(text) > limit else text


def main():
    import chromadb
    from openai import OpenAI

    # ── 加载数据 ──
    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"加载数据: {len(pairs)} 条 Q&A\n")

    # ── 自定义 Embedding：SiliconFlow Qwen3-Embedding-4B（2560维） ──
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    api_base = os.environ.get("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")

    client = OpenAI(api_key=api_key, base_url=api_base)
    EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
    EMBED_DIM = 2560

    def embed_texts(texts: list[str]) -> list[list[float]]:
        """批量 embedding，SiliconFlow API"""
        resp = client.embeddings.create(
            model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM,
        )
        return [item.embedding for item in resp.data]

    # ── 构建 ChromaDB 索引 ──
    palace_path = str(ROOT / "benchmark" / "_mempalace_demo")
    if os.path.exists(palace_path):
        shutil.rmtree(palace_path, ignore_errors=True)

    chroma_client = chromadb.PersistentClient(path=palace_path)
    collection = chroma_client.create_collection(
        name="demo",
        metadata={"hnsw:space": "cosine"},
    )

    texts = [memory_text(p) for p in pairs]
    ids = [f"mem_{i:04d}" for i in range(len(pairs))]
    metadatas = [{"idx": i} for i in range(len(pairs))]

    t0 = time.time()
    # 分批 embedding（API 单次限制）
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = embed_texts(batch)
        all_embeddings.extend(embs)

    collection.add(
        documents=texts, ids=ids, metadatas=metadatas,
        embeddings=all_embeddings,
    )
    build_sec = time.time() - t0
    count = collection.count()
    print(f"构建完成: {build_sec:.1f}s, 存储 {count} 条文档")
    print(f"Embedding: SiliconFlow {EMBED_MODEL} ({EMBED_DIM}维, 中文原生)")
    print(f"检索方式: ChromaDB cosine distance → similarity (1-dist)\n")

    # ── 查询 ──
    top_k = 5
    total_ms = 0
    for qi, query in enumerate(QUERIES, 1):
        # query 也需要 embedding
        q_emb = embed_texts([query])[0]

        t0 = time.time()
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "distances"],
        )
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed

        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]

        print(f"{'─'*70}")
        print(f"  Q{qi}: {query}  [{elapsed:.0f}ms]")
        print(f"{'─'*70}")

        if not docs:
            print("  (空结果)\n")
            continue

        for rank, (doc, dist) in enumerate(zip(docs, dists), 1):
            sim = max(0.0, 1.0 - dist)
            parts = doc.split("\n回答: ", 1)
            q_part = parts[0].replace("用户问: ", "", 1)[:60]
            a_preview = parts[1][:120] + "..." if len(parts) > 1 and len(parts[1]) > 120 else (parts[1] if len(parts) > 1 else "")

            print(f"  #{rank}  sim={sim:.4f}  (dist={dist:.4f})")
            print(f"      问: {q_part}")
            print(f"      答: {a_preview}")
        print()

    avg_ms = total_ms / len(QUERIES)
    print(f"{'═'*70}")
    print(f"  平均查询延迟: {avg_ms:.0f}ms (不含 query embedding 网络耗时)")
    print(f"{'═'*70}")

    # 清理（Windows 可能有文件锁，忽略错误）
    try:
        if os.path.exists(palace_path):
            shutil.rmtree(palace_path, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
