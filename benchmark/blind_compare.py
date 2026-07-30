"""
盲测脚本：三系统（Memento / Mem0 / Letta）跑同一批查询，输出文档里隐藏系统名。

每个系统被随机分配代号 A/B/C，文档里按查询并列展示三列结果，不标哪个是哪个。
代号映射写入单独的 _blind_key.json，等用户盲测完再揭晓。
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.compare_memory_systems import (
    LettaHttpAdapter,
    MementoAdapter,
    Mem0Adapter,
    parse_chat_data,
    memory_text,
    truncate,
)

# 盲测查询：选 6 个，覆盖不同主题
BLIND_QUERIES = [
    "容器启动时配置丢失怎么排查",
    "钢琴连电脑需要什么线和软件",
    "为什么流行歌都是情情爱爱",
    "手机传文件到电脑用什么软件",
    "独立游戏开发者要不要学美术",
    "梯子和局域网冲突怎么解决",
]


def run_all(pairs, top_k=5, limit=None):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results = {}
    adapters = [
        MementoAdapter(top_k, use_concepts=True,
                       embedding_model="api:Qwen/Qwen3-Embedding-4B"),
        Mem0Adapter(top_k, run_id=run_id),
        LettaHttpAdapter(top_k, run_id=run_id),
    ]
    for adapter in adapters:
        print(f"\n=== 构建 {adapter.name} ===", flush=True)
        try:
            build_info = adapter.build(pairs)
            print(f"build OK: stored={build_info.get('stored') or build_info.get('indexed')}", flush=True)
        except Exception as exc:
            print(f"[SKIP] {adapter.name}: {exc}", flush=True)
            results[adapter.name] = {"available": False, "error": str(exc), "queries": {}}
            continue

        query_results = {}
        for q in BLIND_QUERIES:
            try:
                hits, elapsed = adapter.search(q)
                query_results[q] = {
                    "time_ms": round(elapsed * 1000, 1),
                    "hits": [{"text": h.text, "score": h.score} for h in hits],
                }
                print(f"  Q: {q[:30]}... [{elapsed*1000:.0f}ms] {len(hits)} hits", flush=True)
            except Exception as exc:
                query_results[q] = {"error": str(exc), "hits": []}
                print(f"  Q: {q[:30]}... ERROR: {exc}", flush=True)
        results[adapter.name] = {"available": True, "queries": query_results}
    return results, run_id


def write_blind_doc(results, run_id):
    """写盲测文档：系统代号随机，按查询并列，隐藏真名。"""
    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    available_systems = [name for name, r in results.items() if r.get("available")]
    if len(available_systems) < 2:
        print("可用系统不足 2 个，无法盲测")
        return None, None

    # 随机分配代号（换种子，跟上次不同）
    labels = ["A", "B", "C", "D"][:len(available_systems)]
    shuffled = available_systems[:]
    random.Random(run_id).shuffle(shuffled)
    name_to_label = dict(zip(shuffled, labels))
    label_to_name = {v: k for k, v in name_to_label.items()}

    # 写 key 文件（用户盲测完再揭晓）
    key_path = out_dir / f"{run_id}_blind_key.json"
    key_path.write_text(json.dumps(label_to_name, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # 写盲测文档
    lines = [
        "# 记忆系统盲测",
        "",
        f"- 时间：{run_id}",
        f"- 数据：data/testtxt.txt，145 对真实对话",
        f"- 参与系统：{', '.join(labels)}（代号随机，隐藏真名）",
        f"- 每个查询展示各系统的 top-5 结果",
        "",
        "> 盲测说明：不标哪个代号是哪个系统。你读结果，判断哪个代号表现最好。",
        "> 代号对应关系在 `_blind_key.json`，盲测完再看。",
        "",
        "---",
        "",
    ]

    for q in BLIND_QUERIES:
        lines.extend([f"## 查询：{q}", ""])
        # 三列并列：每行一个排名
        max_rank = 5
        # 表头
        header = "| 排名 | " + " | ".join(f"系统 {lab}" for lab in labels) + " |"
        sep = "|------|" + "|".join(["------"] * len(labels)) + "|"
        lines.extend([header, sep])
        for rank in range(max_rank):
            row_cells = []
            for lab in labels:
                sys_name = label_to_name[lab]
                q_result = results[sys_name]["queries"].get(q, {})
                hits = q_result.get("hits", [])
                if rank < len(hits):
                    text = truncate(hits[rank]["text"], 60)
                    row_cells.append(text)
                else:
                    row_cells.append("—")
            lines.append(f"| {rank+1} | " + " | ".join(row_cells) + " |")
        lines.append("")
        # 时间
        time_row = "| 耗时 | " + " | ".join(
            f"{results[label_to_name[lab]]['queries'].get(q, {}).get('time_ms', '—')}ms"
            for lab in labels
        ) + " |"
        lines.extend([time_row, ""])

    md_path = out_dir / f"{run_id}_blind.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, key_path


def main():
    load_dotenv(ROOT / ".env")
    pairs = parse_chat_data(ROOT / "data" / "testtxt.txt")
    print(f"data: {len(pairs)} Q&A")
    print(f"queries: {len(BLIND_QUERIES)}")

    results, run_id = run_all(pairs, top_k=5)

    # 存完整 json 结果（供后续重新渲染，不必重跑）
    out_dir = ROOT / "benchmark" / "results"
    (out_dir / f"{run_id}_blind_raw.json").write_text(
        json.dumps({"run_id": run_id, "queries": BLIND_QUERIES,
                    "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path, key_path = write_blind_doc(results, run_id)

    if md_path:
        print(f"\n{'='*50}")
        print(f"盲测文档：{md_path}")
        print(f"答案 key：{key_path}（盲测完再看）")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
