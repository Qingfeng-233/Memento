"""mem0 快速测试：add + search"""

import os, warnings, shutil, time
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

test_path = str(ROOT / "benchmark" / "_mem0_test")
if os.path.exists(test_path):
    shutil.rmtree(test_path, ignore_errors=True)

from mem0 import Memory

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
            "collection_name": "test_mem0",
            "embedding_model_dims": 2560,
            "path": test_path,
        }
    },
    "version": "v1.1",
}

m = Memory.from_config(config)

# 添加几条
t0 = time.time()
r1 = m.add("我喜欢弹钢琴，用的是Korg D1，连电脑用MIDI线，软件是Pianoteq", user_id="test")
print(f"Add 1 ({time.time()-t0:.1f}s): {r1}")

t0 = time.time()
r2 = m.add("手机传文件到电脑推荐用LocalSend或者Syncthing，不依赖互联网", user_id="test")
print(f"Add 2 ({time.time()-t0:.1f}s): {r2}")

# 搜索
print("\n=== Search: 钢琴连电脑需要什么线和软件 ===")
t0 = time.time()
results = m.search("钢琴连电脑需要什么线和软件", top_k=5, filters={"user_id": "test"})
print(f"  [{(time.time()-t0)*1000:.0f}ms]")
for r in results.get("results", []):
    mem = r.get("memory", "")
    score = r.get("score", "?")
    print(f"  - [{score}] {mem}")

print("\n=== Search: 手机传文件用什么软件 ===")
t0 = time.time()
results = m.search("手机传文件用什么软件", top_k=5, filters={"user_id": "test"})
print(f"  [{(time.time()-t0)*1000:.0f}ms]")
for r in results.get("results", []):
    mem = r.get("memory", "")
    score = r.get("score", "?")
    print(f"  - [{score}] {mem}")

# 清理
shutil.rmtree(test_path, ignore_errors=True)
print("\nDone")
