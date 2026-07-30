from pathlib import Path
import sys, re
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memento.index.keyatten_extractor import MemoryKeywordExtractor

ext = MemoryKeywordExtractor(
    model_path="models/Qwen3-Embedding-0.6B",
    device="cuda", dtype="float16", default_top_k=8)
pairs_text = Path("data/testtxt.txt").read_text(encoding="utf-8-sig")
texts = []
for part in re.split("【用户提问】", pairs_text):
    part = part.strip()
    if not part or "【AI 回答】" not in part:
        continue
    q, a = part.split("【AI 回答】", 1)
    if q.strip() and a.strip():
        texts.append(f"用户问: {q.strip()}\n回答: {a.strip()}")
ext.update_idf(texts)
result = ext.extract(texts[17], top_k=8)
print("qa_0017 最终关键词:", result)
print("引入软件还在吗:", "引入软件" in result)
