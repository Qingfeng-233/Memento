"""溯源：qa_0017 的关键词是怎么抽出来的，「引入软件」怎么进来的。"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def parse(path):
    content = path.read_text(encoding="utf-8-sig")
    pairs = []
    for part in re.split("【用户提问】", content):
        part = part.strip()
        if not part or "【AI 回答】" not in part:
            continue
        q, a = part.split("【AI 回答】", 1)
        if q.strip() and a.strip():
            pairs.append({"q": q.strip(), "a": a.strip()})
    return pairs


def main():
    pairs = parse(ROOT / "data" / "testtxt.txt")
    text = f"用户问: {pairs[17]['q']}\n回答: {pairs[17]['a']}"

    print("=" * 70)
    print("qa_0017 完整文本：")
    print("=" * 70)
    print(text)
    print()

    # 1. keyatten 原始抽取（不加 phrase-merge）
    from memento.index.keyatten_extractor import MemoryKeywordExtractor
    ext = MemoryKeywordExtractor(
        model_path="models/Qwen3-Embedding-0.6B",
        device="cuda", dtype="float16", default_top_k=8,
        phrase_merge_enabled=False,  # 关掉 phrase-merge
    )
    raw_kws = ext.extract(text, top_k=8)
    print("=" * 70)
    print("1. keyatten 原始抽取（phrase_merge 关闭）：")
    print("=" * 70)
    print(f"  {raw_kws}")

    # 2. 开启 phrase-merge
    ext2 = MemoryKeywordExtractor(
        model_path="models/Qwen3-Embedding-0.6B",
        device="cuda", dtype="float16", default_top_k=8,
        phrase_merge_enabled=True,
    )
    merged_kws = ext2.extract(text, top_k=8)
    print()
    print("=" * 70)
    print("2. keyatten + phrase_merge 开启：")
    print("=" * 70)
    print(f"  {merged_kws}")

    # 3. 对比：「引入软件」是从哪来的
    print()
    print("=" * 70)
    print("3. 分析：「引入软件」是怎么来的")
    print("=" * 70)
    only_in_merged = [kw for kw in merged_kws if kw not in raw_kws]
    print(f"  phrase_merge 新增的词：{only_in_merged}")
    for kw in only_in_merged:
        # 看 jieba 怎么切的
        import jieba.posseg as pseg
        tokens = [(w.word, w.flag) for w in pseg.cut(text)]
        print(f"\n  「{kw}」在文本中的 jieba 分词上下文：")
        for i, (word, flag) in enumerate(tokens):
            if "引入" in word or "软件" in word or "确实" in word:
                context = tokens[max(0,i-2):i+3]
                ctx_str = " | ".join(f"{w}({f})" for w, f in context)
                print(f"    位置 {i}: {ctx_str}")


if __name__ == "__main__":
    main()
