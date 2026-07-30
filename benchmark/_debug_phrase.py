from memento.index.keyatten_extractor import MemoryKeywordExtractor
import jieba.posseg as pseg

ext = MemoryKeywordExtractor(
    model_path="models/Qwen3-Embedding-0.6B",
    device="cuda", dtype="float16", default_top_k=8)

text = "引入软件确实是最好的办法"
tokens = [(w.word, w.flag) for w in pseg.cut(text)]
print("jieba tokens:", tokens)
print()
for w, f in tokens:
    in_stop = w in ext.PHRASE_MERGE_STOP_TOKENS
    is_content = ext._is_content_token(w, f)
    print(f"  {w} ({f})  content={is_content}  stop={in_stop}")

# 模拟窗口构建
print("\n窗口构建过程:")
window = []
windows = []
for w, f in tokens:
    if w in ext.PHRASE_MERGE_STOP_TOKENS:
        print(f"  → STOP: '{w}' 截断窗口 window={window}")
        windows.append(window[:])
        window = []
        continue
    if ext._is_content_token(w, f):
        window.append(w)
        print(f"  → content: '{w}' 加入窗口 window={window}")
        continue
    print(f"  → non-content: '{w}' 截断窗口 window={window}")
    windows.append(window[:])
    window = []
windows.append(window[:])

print(f"\n最终窗口列表: {windows}")
for i, win in enumerate(windows):
    if len(win) >= 2:
        phrases = ext._phrases_from_token_window(win, {"软件"})
        print(f"  窗口 {i} {win} → phrases: {phrases}")
