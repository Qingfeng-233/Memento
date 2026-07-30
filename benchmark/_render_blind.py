"""把盲测 raw json 渲染成可读 txt（完整文本，不截断）。"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
run_id = "20260614-193929"
raw = json.loads(
    (ROOT / "benchmark" / "results" / f"{run_id}_blind_raw.json").read_text(encoding="utf-8")
)
results = raw["results"]
queries = raw["queries"]

available = [n for n, r in results.items() if r.get("available")]
rng = random.Random(run_id)
shuffled = available[:]
rng.shuffle(shuffled)
labels = ["A", "B", "C"][: len(shuffled)]
name_to_label = dict(zip(shuffled, labels))
label_to_name = {v: k for k, v in name_to_label.items()}

out = []
out.append("=" * 90)
out.append("记忆系统盲测结果")
out.append("数据：data/testtxt.txt，145 对真实对话")
out.append("系统：" + ", ".join(labels) + "（代号随机，隐藏真名）")
out.append("说明：每条记忆给完整文本。判断完 A/B/C 谁最好，再看 _blind_key.json")
out.append("=" * 90)

for qi, q in enumerate(queries, 1):
    out.append("")
    out.append("#" * 90)
    out.append(f"  查询 {qi}：{q}")
    out.append("#" * 90)
    for rank in range(5):
        out.append("")
        out.append(f"  ── 排名 {rank + 1} ──")
        for lab in labels:
            sn = label_to_name[lab]
            qres = results[sn]["queries"].get(q, {})
            hits = qres.get("hits", [])
            time_ms = qres.get("time_ms", "—")
            if rank < len(hits):
                text = hits[rank]["text"].strip()
                score = hits[rank].get("score")
                score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
                out.append(f"  系统 {lab} [score={score_s}]")
                out.append(f"    {text}")
            else:
                out.append(f"  系统 {lab}: —")
    times = "  ".join(
        f"{lab}={results[label_to_name[lab]]['queries'].get(q, {}).get('time_ms', '—')}ms"
        for lab in labels
    )
    out.append("")
    out.append(f"  耗时：{times}")

out.append("")
out.append("=" * 90)
out.append("盲测完请查看 benchmark/results/20260614-192149_blind_key.json 揭晓代号")
out.append("=" * 90)

txt = "\n".join(out)
dst = ROOT / "benchmark" / "results" / f"{run_id}_blind_readable.txt"
dst.write_text(txt, encoding="utf-8")
print(f"写入 {dst}")
print(f"长度 {len(txt)} 字符")
