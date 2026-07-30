"""追踪：extract() 内部，keywords 和 merged 分别是什么，子串抑制为什么没拦住。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memento.index.keyatten_extractor import MemoryKeywordExtractor

ext = MemoryKeywordExtractor(
    model_path="models/Qwen3-Embedding-0.6B",
    device="cuda", dtype="float16", default_top_k=8,
    phrase_merge_enabled=True,
)

# 模拟 build_concept_graph 的环境：需要先 update_idf
pairs_text = Path("data/testtxt.txt").read_text(encoding="utf-8-sig")
import re
texts = []
for part in re.split("【用户提问】", pairs_text):
    part = part.strip()
    if not part or "【AI 回答】" not in part:
        continue
    q, a = part.split("【AI 回答】", 1)
    if q.strip() and a.strip():
        texts.append(f"用户问: {q.strip()}\n回答: {a.strip()}")
ext.update_idf(texts)

# qa_0017 的文本
qa17 = texts[17]
print("qa_0017 text:", qa17[:100], "...\n")

# 拆开 extract 的步骤
method = "fusion_attn_idf"
keywords = ext._extractor.extract_keywords(qa17, method=method, top_k=8, idf_lookup=ext._idf_lookup)
print("1. raw keyatten (with IDF):", keywords)

merged = ext._merge_adjacent_phrases(qa17, keywords)
print("2. merged phrases:", merged)

suppressed = ext._suppress_junk_phrases(merged, keywords)
print("3. after suppress:", suppressed)

print("4. keywords that are substrings of merged:")
raw_set = set(keywords)
for phrase in merged:
    for rk in raw_set:
        if rk in phrase and rk != phrase and len(rk) >= 2:
            print(f"   '{rk}' is substring of '{phrase}' → should be suppressed")
