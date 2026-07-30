from pathlib import Path
import json, hashlib, re

content = Path("data/testtxt.txt").read_text(encoding="utf-8-sig")
pairs = []
for part in re.split("【用户提问】", content):
    part = part.strip()
    if not part or "【AI 回答】" not in part:
        continue
    q, a = part.split("【AI 回答】", 1)
    if q.strip() and a.strip():
        pairs.append({"q": q.strip(), "a": a.strip()})

qa17_text = f"用户问: {pairs[17]['q']}\n回答: {pairs[17]['a']}"
print("qa_0017 text (first 120):", qa17_text[:120])

key = hashlib.sha256(qa17_text.encode("utf-8")).hexdigest()
cache_path = Path("data/surprisal_cache") / f"{key}.json"
print(f"cache key: {key[:20]}...")
print(f"cache exists: {cache_path.exists()}")

if cache_path.exists():
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"cached keywords: {list(data.keys())[:10]}")
    for kw in ["引入软件", "引入软件确实", "发现钢琴", "钢琴型号"]:
        if kw in data:
            d = data[kw]
            print(f"  {kw}: first={d['first']:.3f} max={d['max']:.3f}")
        else:
            print(f"  {kw}: NOT in cache")
else:
    print("CACHE MISS — qa_0017 的 surprisal 没被算过！")
    print("这就是边权=1.0 的原因——surprisal 没生效")
