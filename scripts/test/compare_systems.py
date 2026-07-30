"""
Memento vs Mem0 vs Letta vs Zep 对比测试

所有系统使用相同的:
  - 测试数据 (data/testtxt.txt, 146 Q&A pairs)
  - Embedding: Qwen3-Embedding-4B (2560-dim, last-token pooling)
  - 查询集: 8 个手工 query

公平性:
  - Mem0 用 infer=False 直接存原文（不经 LLM fact extraction）
  - Memento 用相同的 embedding 模型
  - Letta/Zep 需要服务器，自动检测可用性
"""

import sys, re, time, json, os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/test -> project root
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# 禁用 Mem0 遥测，避免初始化卡住
os.environ["MEM0_TELEMETRY"] = "false"


def gpu_cleanup(obj=None):
    """轻量清理（API 模式下基本不需要）"""
    import gc
    gc.collect()


# ─── 数据解析 ─────────────────────────────────────────────

def parse_chat_data(filepath):
    """解析 testtxt.txt 中的 Q&A 对"""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()
    parts = re.split(r'【用户提问】', content)
    pairs = []
    for part in parts:
        part = part.strip()
        if not part or '【AI 回答】' not in part:
            continue
        q, a = part.split('【AI 回答】', 1)
        q, a = q.strip(), a.strip()
        if q and a:
            pairs.append({"question": q, "answer": a})
    return pairs


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


def truncate(text, n=80):
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


# ─── Memento ─────────────────────────────────────────────

def test_memento(pairs, queries, top_k=5):
    """Memento: 向量检索 + 图扩散（SiliconFlow API embedding）"""
    print("\n" + "=" * 70)
    print("  MEMENTO (Qwen3-Embedding-4B via API + 关键词图扩散)")
    print("=" * 70)

    from memento.api import Memento

    t0 = time.time()
    m = Memento(
        embedding_model="api:Qwen/Qwen3-Embedding-4B",
        diffusion_hops=2,
        diffusion_alpha=0.3,
        diffusion_beta=0.6,
    )

    # 添加节点（只存 AI 回答）
    for i, pair in enumerate(pairs):
        m.add_node(
            text=pair["answer"],
            node_id=f"m_{i:04d}",
            importance=0.5,
            tags=[f"q:{truncate(pair['question'], 30)}"],
        )
    m.build_index()
    t_build = time.time() - t0
    print(f"  索引构建: {t_build:.1f}s ({len(pairs)} nodes)")

    # 关键词建边
    t0 = time.time()
    kw_result = m.build_keyword_edges(
        top_k=5, min_overlap=1, max_node_freq=30,
        weight_per_keyword=0.15,
    )
    t_kw = time.time() - t0
    print(f"  关键词建边: {t_kw:.1f}s "
          f"(edges={kw_result['edges_added']}, "
          f"vocab={kw_result['vocab_size']}, "
          f"total_kw={kw_result['total_keywords']})")

    # 查询
    results = {}
    for q in queries:
        t0 = time.time()
        hits = m.query(q, k=top_k, seed_k=20)
        t_q = time.time() - t0
        results[q] = {"hits": hits, "time": t_q}
        print(f"\n  Q: {q}")
        print(f"  [{t_q*1000:.0f}ms]")
        for i, h in enumerate(hits):
            print(f"    {i+1}. [{h['score']:.4f}] {truncate(h['text'])}")

    stats = m.stats
    print(f"\n  Stats: nodes={stats['total_nodes']}, "
          f"edges={stats['total_edges']}, "
          f"kw_nodes={stats['keyword_nodes']}")

    return results


# ─── Mem0 ────────────────────────────────────────────────

def test_mem0(pairs, queries, top_k=5):
    """Mem0: 向量存储 + 搜索（SiliconFlow API embedding + 直接存储）"""
    print("\n" + "=" * 70)
    print("  MEM0 (Qwen3-Embedding-4B via API + 直接存储)")
    print("=" * 70)

    from mem0 import Memory

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "deepseek-v4-flash",
                "api_key": os.getenv("OPENCODE_API_KEY"),
                "openai_base_url": os.getenv("OPENCODE_API_BASE"),
                "temperature": 0.0,
                "max_tokens": 1000,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "Qwen/Qwen3-Embedding-4B",
                "api_key": os.getenv("SILICONFLOW_API_KEY"),
                "openai_base_url": os.getenv("SILICONFLOW_API_BASE"),
                "embedding_dims": 2560,
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "memento_compare",
                "embedding_model_dims": 2560,
                "on_disk": True,
                "path": str(ROOT / "scripts" / "test" / "_mem0_qdrant").replace("\\", "/"),
            }
        },
    }

    t0 = time.time()
    m = Memory.from_config(config)
    t_init = time.time() - t0
    print(f"  初始化: {t_init:.1f}s", flush=True)

    # 添加数据（infer=False 直接存原文，不经 LLM 提取）
    t0 = time.time()
    added = 0
    errors = 0
    for i, pair in enumerate(pairs):
        try:
            m.add(
                pair["answer"],
                user_id="test_user",
                infer=False,
                metadata={"idx": i},
            )
            added += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                    print(f"  [WARN] add #{i} failed: {e}")
        if (i + 1) % 20 == 0 or i == len(pairs) - 1:
            print(f"  添加进度: {added}/{len(pairs)} (errors={errors})", flush=True)
    t_add = time.time() - t0
    print(f"  数据添加: {t_add:.1f}s ({added} stored, {errors} errors)", flush=True)

    # 搜索（Mem0 v2 需要用 filters 而非直接传 user_id）
    results = {}
    for q in queries:
        t0 = time.time()
        try:
            hits_raw = m.search(q, filters={"user_id": "test_user"}, top_k=top_k)
            t_q = time.time() - t0
            hits = []
            for h in (hits_raw.get("results", []) if isinstance(hits_raw, dict) else hits_raw):
                text = h.get("memory", "") or h.get("text", "")
                score = h.get("score", 0.0)
                hits.append({"text": text, "score": score})
            results[q] = {"hits": hits, "time": t_q}
            print(f"\n  Q: {q}")
            print(f"  [{t_q*1000:.0f}ms]")
            for i, h in enumerate(hits):
                print(f"    {i+1}. [{h['score']:.4f}] {truncate(h['text'])}")
        except Exception as e:
            print(f"\n  Q: {q} -> ERROR: {e}")
            results[q] = {"hits": [], "time": 0, "error": str(e)}

    return results


# ─── Letta ───────────────────────────────────────────────

def test_letta(pairs, queries, top_k=5):
    """Letta: 需要运行 letta server，自动检测"""
    print("\n" + "=" * 70)
    print("  LETTA (MemGPT)")
    print("=" * 70)

    import requests
    # 尝试检测 letta server
    for port in [8283, 8083, 8284]:
        try:
            r = requests.get(f"http://localhost:{port}/v1/health", timeout=2)
            if r.status_code == 200:
                print(f"  Letta server found at :{port}")
                break
        except:
            continue
    else:
        print("  [SKIP] 没有检测到 Letta server")
        print("  启动方式: letta server")
        print("  Letta 是全 agent 框架，需要 server 运行才能测试 memory")
        return None

    # 如果有 server，执行测试
    from letta import RESTClient
    client = RESTClient(base_url=f"http://localhost:{port}")
    # ... 后续测试逻辑（如果 server 存在）
    return {"note": "Letta server found but test not fully implemented"}


# ─── Zep ─────────────────────────────────────────────────

def test_zep(pairs, queries, top_k=5):
    """Zep: 需要运行 Zep server (Docker)，自动检测"""
    print("\n" + "=" * 70)
    print("  ZEP (Graphiti)")
    print("=" * 70)

    import requests
    try:
        r = requests.get("http://localhost:8000/api/v2/healthz", timeout=2)
        if r.status_code == 200:
            print("  Zep server found")
    except:
        pass

    # 尝试其他端口
    for port in [8000, 8080, 9000]:
        try:
            r = requests.get(f"http://localhost:{port}/api/v2/healthz", timeout=2)
            if r.status_code == 200:
                print(f"  Zep server found at :{port}")
                break
        except:
            continue
    else:
        print("  [SKIP] 没有检测到 Zep server")
        print("  启动方式: docker compose up (需要 Zep Docker)")
        print("  Zep 是时序知识图谱系统，需要 server 运行")
        return None

    return {"note": "Zep server found but test not fully implemented"}


# ─── 对比汇总 ─────────────────────────────────────────────

def print_comparison(all_results):
    """打印各系统对比汇总"""
    print("\n" + "=" * 70)
    print("  对比汇总")
    print("=" * 70)

    systems = [k for k, v in all_results.items() if v is not None]
    if not systems:
        print("  没有可用的系统结果")
        return

    for q in QUERIES:
        print(f"\n  Q: {q}")
        print(f"  {'─' * 60}")
        for sys_name in systems:
            r = all_results[sys_name].get(q, {})
            hits = r.get("hits", [])
            t = r.get("time", 0)
            err = r.get("error")
            print(f"  [{sys_name}] ({t*1000:.0f}ms, {len(hits)} hits)")
            if err:
                print(f"    ERROR: {err}")
            for i, h in enumerate(hits[:3]):
                score = h.get("score", 0)
                text = truncate(h.get("text", ""), 70)
                print(f"    {i+1}. [{score:.4f}] {text}")
        print()


# ─── 主入口 ───────────────────────────────────────────────

def main():
    data_path = ROOT / "data" / "testtxt.txt"
    pairs = parse_chat_data(str(data_path))
    print(f"测试数据: {len(pairs)} Q&A pairs")
    print(f"查询集: {len(QUERIES)} queries")

    all_results = {}

    # 1. Memento
    try:
        all_results["Memento"] = test_memento(pairs, QUERIES)
    except Exception as e:
        print(f"  [FAIL] Memento: {e}")
        import traceback; traceback.print_exc()
        all_results["Memento"] = None
    finally:
        gpu_cleanup()

    # 2. Mem0
    try:
        all_results["Mem0"] = test_mem0(pairs, QUERIES)
    except Exception as e:
        print(f"  [FAIL] Mem0: {e}")
        import traceback; traceback.print_exc()
        all_results["Mem0"] = None
    finally:
        gpu_cleanup()

    # 3. Letta
    try:
        all_results["Letta"] = test_letta(pairs, QUERIES)
    except Exception as e:
        print(f"  [FAIL] Letta: {e}")
        all_results["Letta"] = None

    # 4. Zep
    try:
        all_results["Zep"] = test_zep(pairs, QUERIES)
    except Exception as e:
        print(f"  [FAIL] Zep: {e}")
        all_results["Zep"] = None

    # 汇总
    print_comparison(all_results)

    # 保存结果
    output_path = ROOT / "scripts" / "test" / "compare_systems_output.txt"
    import io
    buf = io.StringIO()
    # 简单 redirect 方式：直接写文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 记忆系统对比测试结果\n")
        f.write(f"# 日期: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# 数据: {len(pairs)} Q&A pairs\n")
        f.write(f"# Embedding: Qwen3-Embedding-4B (2560-dim)\n\n")
        for sys_name, results in all_results.items():
            f.write(f"\n## {sys_name}\n")
            if results is None:
                f.write("  [NOT AVAILABLE]\n")
                continue
            for q, r in results.items():
                f.write(f"\nQ: {q}\n")
                f.write(f"  Time: {r.get('time', 0)*1000:.0f}ms\n")
                for i, h in enumerate(r.get("hits", [])):
                    f.write(f"  {i+1}. [{h.get('score', 0):.4f}] "
                            f"{truncate(h.get('text', ''), 100)}\n")
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
