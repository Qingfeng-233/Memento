from pathlib import Path
d = Path("data/background_idf")
for f in ["cn_stopwords.txt", "hit_stopwords.txt", "baidu_stopwords.txt"]:
    p = d / f
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{f}: {len(lines)} words, first 8: {lines[:8]}")
    for w in ["usb", "USB", "软件", "开心", "难过", "电脑", "手机", "喜欢"]:
        if w in lines:
            print(f"  HIT: {w}")
print("---合并去重---")
allw = set()
for f in ["cn_stopwords.txt", "hit_stopwords.txt", "baidu_stopwords.txt"]:
    for l in (d / f).read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l:
            allw.add(l)
print(f"合并去重: {len(allw)} words")
