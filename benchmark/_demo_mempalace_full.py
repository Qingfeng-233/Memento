"""MemPalace 完整功能演示：miner + drawer + closet + BM25 混合搜索 + closet boost

使用 SiliconFlow Qwen3-Embedding-4B 作为 embedding（monkey-patch）
"""

import os, re, sys, time, shutil, warnings, threading
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
    from openai import OpenAI

    # ── SiliconFlow API 客户端 ──
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    api_base = os.environ.get("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")
    sf_client = OpenAI(api_key=api_key, base_url=api_base)
    EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
    EMBED_DIM = 2560

    # ── Monkey-patch MemPalace 的 embedding 函数 ──
    # 让 miner、closet 生成、search 全部使用 Qwen3-Embedding-4B
    class SiliconFlowEmbeddingFunction:
        """ChromaDB 兼容的 embedding 函数，调用 SiliconFlow API"""
        _instance = None
        _lock = threading.Lock()

        def __new__(cls, *args, **kwargs):
            # 单例，避免重复创建 API 客户端
            if cls._instance is None:
                with cls._lock:
                    if cls._instance is None:
                        cls._instance = super().__new__(cls)
                        cls._instance._initialized = False
            return cls._instance

        def __init__(self, *args, **kwargs):
            if self._initialized:
                return
            self._initialized = True
            self._client = sf_client
            self._model = EMBED_MODEL
            self._dim = EMBED_DIM
            self._batch_size = 32

        @staticmethod
        def name() -> str:
            return "siliconflow_qwen3_4b"

        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            if not input:
                return []
            all_embeddings = []
            for i in range(0, len(input), self._batch_size):
                batch = input[i:i + self._batch_size]
                resp = self._client.embeddings.create(
                    model=self._model, input=batch, dimensions=self._dim,
                )
                all_embeddings.extend([item.embedding for item in resp.data])
            return all_embeddings

        def embed_query(self, input):
            return self(input)

        def embed_documents(self, input):
            return self(input)

    # patch embedding 模块
    import mempalace.embedding as _emb
    _original_get_ef = _emb.get_embedding_function
    _siliconflow_ef = SiliconFlowEmbeddingFunction()
    _emb._EF_CACHE.clear()

    def _patched_get_ef(device=None, model=None):
        return _siliconflow_ef

    _emb.get_embedding_function = _patched_get_ef

    # ── 准备项目目录 ──
    project_dir = ROOT / "benchmark" / "_mempalace_project"
    palace_path = str(ROOT / "benchmark" / "_mempalace_palace")

    if project_dir.exists():
        shutil.rmtree(project_dir, ignore_errors=True)
    if os.path.exists(palace_path):
        shutil.rmtree(palace_path, ignore_errors=True)

    project_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据并按主题分文件
    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"加载数据: {len(pairs)} 条 Q&A\n")

    # 全部写入一个文件，让 miner 自动 chunk
    all_text = []
    for i, pair in enumerate(pairs):
        all_text.append(f"【对话 #{i+1}】")
        all_text.append(memory_text(pair))
        all_text.append("")  # 空行分隔

    (project_dir / "conversations.txt").write_text(
        "\n".join(all_text), encoding="utf-8"
    )

    # 创建 mempalace.yaml
    yaml_content = """wing: "memento-benchmark"
rooms:
  - name: "conversations"
    description: "AI 对话记录"
    keywords: ["对话", "问答", "聊天"]
"""
    (project_dir / "mempalace.yaml").write_text(yaml_content, encoding="utf-8")

    # ── 运行 Miner（生成 drawer + closet）──
    from mempalace.miner import mine

    print("=" * 60)
    print("  MemPalace Miner: 入库中...")
    print("=" * 60)

    t0 = time.time()
    mine(
        project_dir=str(project_dir),
        palace_path=palace_path,
        respect_gitignore=False,
    )
    mine_sec = time.time() - t0

    print(f"\nMiner 完成: {mine_sec:.1f}s")

    # 查看 drawer 和 closet 数量
    import chromadb
    chroma_client = chromadb.PersistentClient(path=palace_path)
    collections = chroma_client.list_collections()
    print(f"Collections: {[c.name for c in collections]}")

    for col in collections:
        print(f"  {col.name}: {col.count()} 条记录")

    # ── 搜索（使用 search_memories 完整管线）──
    from mempalace.searcher import search_memories

    print(f"\n{'=' * 60}")
    print("  MemPalace search_memories: 完整管线")
    print("  (向量召回 + BM25 重排 + closet boost)")
    print(f"{'=' * 60}\n")

    top_k = 5
    total_ms = 0
    for qi, query in enumerate(QUERIES, 1):
        t0 = time.time()
        result = search_memories(
            query=query,
            palace_path=palace_path,
            n_results=top_k,
            candidate_strategy="union",  # 向量 + BM25 联合候选
        )
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed

        print(f"{'─' * 70}")
        print(f"  Q{qi}: {query}  [{elapsed:.0f}ms]")
        print(f"{'─' * 70}")

        if "error" in result:
            print(f"  ERROR: {result['error']}\n")
            continue

        hits = result.get("results", [])
        if not hits:
            print("  (空结果)\n")
            continue

        for rank, hit in enumerate(hits[:top_k], 1):
            text = hit.get("text", "")
            score = hit.get("score", hit.get("final_score", None))
            bm25 = hit.get("bm25_score", 0)
            closet_boost = hit.get("closet_boost", 0)
            source = hit.get("metadata", {}).get("source_file", "?")

            parts = text.split("\n回答: ", 1)
            q_part = parts[0].replace("用户问: ", "", 1)[:60]
            a_preview = parts[1][:120] + "..." if len(parts) > 1 and len(parts[1]) > 120 else (parts[1] if len(parts) > 1 else "")

            score_str = f"{score:.4f}" if score is not None else "-"
            print(f"  #{rank}  score={score_str}  bm25={bm25}  closet_boost={closet_boost}")
            print(f"      source: {source}")
            print(f"      问: {q_part}")
            print(f"      答: {a_preview}")
        print()

    avg_ms = total_ms / len(QUERIES)
    print(f"{'═' * 60}")
    print(f"  平均查询延迟: {avg_ms:.0f}ms")
    print(f"{'═' * 60}")

    # 还原 embedding
    _emb.get_embedding_function = _original_get_ef

    # 清理
    try:
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)
        if os.path.exists(palace_path):
            shutil.rmtree(palace_path, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
