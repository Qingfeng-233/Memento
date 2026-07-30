"""验证 LLM surprisal：file 低 / LocalSend 高 / 钢琴在钢琴对话里中等。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memento.engine.surprisal_calculator import SurprisalCalculator

calc = SurprisalCalculator(model_path="models/Qwen/Qwen3.5-4B")

# 测试文本 + 关键词
tests = [
    {
        "text": "用户问: 这玩意能买吗？ Attached 1 file. - 1000002638.jpg 回答: 这款完全可以买，而且是个加量不加价的惊喜发现。这比我刚才推荐给你的那个 AMC-10 还要好。",
        "keywords": ["file", "MIDI", "买", "AMC"],
    },
    {
        "text": "用户问: 我爸提醒之下，我发现钢琴有些荒废了。解构一下，发现就是老是弹错音，导致弹出来的就没那种感觉。",
        "keywords": ["钢琴", "弹错音", "荒废", "软件"],
    },
    {
        "text": "用户问: 有没有手机上一划就能上传到图片到电脑上的软件？感觉不便捷，你知道吧？回答: 推荐 LocalSend 或 Syncthing。",
        "keywords": ["LocalSend", "Syncthing", "手机", "软件", "电脑"],
    },
    {
        "text": "用户问: 我还以为电脑梯子问题，结果发现v2rayN默认绕过局域网，原来手机上NekoBox的锅。回答: 破案了！居然是 NekoBox。",
        "keywords": ["v2rayN", "NekoBox", "局域网", "梯子", "电脑"],
    },
    {
        "text": "用户问: 只是想要无需外部依赖的本地配置。回答: 这是一句非常赤裸且让人动容的话。去掉了一切哲学包装。",
        "keywords": ["爱意", "赤裸", "哲学", "动容"],
    },
]

for i, t in enumerate(tests):
    print(f"\n{'='*70}")
    print(f"文本 {i+1}: {t['text'][:60]}...")
    print(f"{'='*70}")
    results = calc.compute(t["text"], t["keywords"])
    if not results:
        print("  （无结果）")
        continue
    # 按 surprisal 排序
    sorted_kw = sorted(results.items(), key=lambda x: x[1]["first"])
    print(f"  {'关键词':<14} {'first':>7} {'max':>7} {'count':>5}  判断")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*5}")
    for kw, info in sorted_kw:
        s = info["first"]
        # 粗略判断
        if s < 2:
            judge = "低（不意外，预期内）"
        elif s < 5:
            judge = "中（有点意外）"
        else:
            judge = "高（很意外/信息量大）"
        print(f"  {kw:<14} {s:>7.3f} {info['max']:>7.3f} {info['count']:>5}  {judge}")

print("\n" + "=" * 70)
print("验证关键点：")
print("  file 在 'Attached 1 file' 里 → first 应该低（预期内）")
print("  LocalSend 在手机软件推荐里 → first 应该高（专有名词）")
print("  钢琴在钢琴荒废讨论里 → first 应该低（主题词）")
print("  荒废/弹错音 → first 应该高（具体描述）")
print("=" * 70)
