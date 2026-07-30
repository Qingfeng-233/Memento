"""
深层探针：直接用 embedding 编码若干代表性短词，看两两余弦矩阵。

目的：搞清楚 embedding 模型对短关键词的区分度到底有多差。
不是在概念图层面看（那有边权/energy 干扰），而是回到最原始的向量空间。
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import numpy as np


def main():
    from memento.api import Memento

    # 只用 VectorIndex，不建图
    engine = Memento(embedding_model="api:Qwen/Qwen3-Embedding-4B")
    # 假装建索引（只需编码能力）
    engine.add_node("dummy", node_id="d0")
    engine.build_index()

    # 代表性词组：
    # A. 手机传文件域
    # B. 钢琴MIDI域
    # C. 梯子网络域
    # D. 容器配置域
    # E. 通用短英文/缩写
    # F. 完整句子（对比）
    words = [
        # A 手机传文件
        "LocalSend", "Syncthing", "手机传文件", "电脑扫描手机",
        # B 钢琴MIDI
        "Korg D1", "MIDI", "MIDI to USB", "钢琴连电脑", "音频MIDI",
        # C 梯子网络
        "v2rayN", "局域网", "梯子", "网络冲突",
        # D 情感容器配置
        "容器配置", "孤独", "羁绊", "Docker Compose",
        # E 通用短英文/缩写（疑似噪声源）
        "USB", "file", "软件", "电脑", "手机", "线",
        # F 完整句子（看长文本的区分度对比）
        "手机传文件到电脑用什么软件",
        "钢琴连电脑需要什么线和软件",
    ]

    vecs = engine.vector_index.encode(words, mode="document")
    # 归一化
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_n = vecs / norms

    # 两两余弦矩阵
    sim = vecs_n @ vecs_n.T

    # 打印矩阵（只打印 > 0.3 的，避免噪声）
    print("两两余弦矩阵（只显示 > 0.30 的值，空白 = < 0.30）\n")
    # 表头
    short = [w[:8] for w in words]
    print("                  " + " ".join(f"{s:>8}" for s in short))
    for i, w in enumerate(words):
        row = []
        for j in range(len(words)):
            v = sim[i, j]
            if i == j:
                row.append("   1.00")
            elif v > 0.30:
                row.append(f"{v:8.3f}")
            else:
                row.append("       .")
        print(f"{w[:16]:<16} " + " ".join(row))

    # 重点跨域对
    print(f"\n{'='*70}")
    print("跨域重点对（应该低，如果高就是 embedding 问题）")
    print(f"{'='*70}")
    pairs = [
        ("LocalSend", "Korg D1"),
        ("LocalSend", "MIDI"),
        ("Syncthing", "Korg D1"),
        ("Syncthing", "MIDI"),
        ("USB", "LocalSend"),
        ("USB", "v2rayN"),
        ("USB", "容器配置"),
        ("file", "MIDI"),
        ("file", "LocalSend"),
        ("软件", "钢琴连电脑"),
        ("线", "容器配置"),
        ("电脑", "手机"),
    ]
    w2i = {w: i for i, w in enumerate(words)}
    for a, b in pairs:
        if a in w2i and b in w2i:
            cos = float(sim[w2i[a], w2i[b]])
            flag = "⚠️异常高" if cos > 0.5 else ("偏高" if cos > 0.35 else "正常")
            print(f"  {a:<12} vs {b:<12}  cos={cos:.4f}  {flag}")

    # 域内对比（应该高）
    print(f"\n{'='*70}")
    print("域内对比（应该高，如果低说明模型连同域都分不清）")
    print(f"{'='*70}")
    intra = [
        ("LocalSend", "Syncthing"),
        ("手机传文件", "电脑扫描手机"),
        ("Korg D1", "MIDI"),
        ("Korg D1", "钢琴连电脑"),
        ("v2rayN", "局域网"),
    ]
    for a, b in intra:
        if a in w2i and b in w2i:
            cos = float(sim[w2i[a], w2i[b]])
            flag = "正常" if cos > 0.4 else "⚠️异常低"
            print(f"  {a:<12} vs {b:<12}  cos={cos:.4f}  {flag}")


if __name__ == "__main__":
    main()
