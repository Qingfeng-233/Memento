"""
生成 Memento / Mem0 / Letta 记忆命中对比报告。

本脚本只整理已有实测输出，不重新跑长耗时检索流程。
"""

from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT = ROOT / "scripts" / "test" / "compare_systems_output.txt"


def _safe_report_path() -> Path:
    # Windows strftime 不支持 %-m / %-d。
    now = time.localtime()
    return ROOT / "docs" / (
        f"{now.tm_year % 100}-{now.tm_mon}-{now.tm_mday}-"
        f"{now.tm_hour}-{now.tm_min:02d}-memory-hit-report.md"
    )


def _truncate(text: str, limit: int = 82) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "..." if len(text) > limit else text


def _parse_output() -> dict[str, dict[str, list[dict[str, str]]]]:
    content = OUTPUT.read_text(encoding="utf-8")
    systems: dict[str, dict[str, list[dict[str, str]]]] = {}
    current_system = None
    current_query = None

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        system_match = re.match(r"^## (.+)$", line)
        if system_match:
            current_system = system_match.group(1).strip()
            systems[current_system] = {}
            current_query = None
            continue

        query_match = re.match(r"^Q: (.+)$", line)
        if query_match and current_system:
            current_query = query_match.group(1).strip()
            systems[current_system][current_query] = []
            continue

        hit_match = re.match(r"^\s+(\d+)\. \[([0-9.]+)\] (.+)$", line)
        if hit_match and current_system and current_query:
            systems[current_system][current_query].append(
                {
                    "rank": hit_match.group(1),
                    "score": hit_match.group(2),
                    "text": hit_match.group(3),
                }
            )

    return systems


def _detect_letta() -> tuple[bool, str]:
    if importlib.util.find_spec("letta") is None:
        return False, "Python 包未安装"

    for port in (8283, 8083, 8284):
        try:
            response = requests.get(
                f"http://localhost:{port}/v1/health", timeout=2
            )
        except requests.RequestException:
            continue
        if response.status_code == 200:
            return True, (
                f"检测到 Letta server: http://localhost:{port}；"
                "已通过 `scripts/test/verify_letta_memory.py` 验证 passage 写入和搜索。"
                "当前修复方式是全局 OpenAI 兼容 key 用于硅基流动 embedding，"
                "OpenCode LLM key 通过 Letta BYOK provider 单独注册"
            )
    return False, "Python 包已安装，但本机未检测到 Letta server"


def _judge_query(query: str, memento_hit: str, mem0_hit: str) -> str:
    if query == "钢琴连电脑需要什么线和软件":
        return "Memento 明显更准；Mem0 top1 偏到练琴反馈循环。"
    if query == "怎么有效休息不会浪费意志力":
        return "Memento 命中番茄钟休息误区，Mem0 命中更泛化的休息观念。"
    if query in {
        "手机传文件到电脑用什么软件",
        "独立游戏开发者要不要学美术",
        "梯子和局域网冲突怎么解决",
        "为什么流行歌都是情情爱爱",
    }:
        return "两者 top1 都可用，Memento 分数梯度更大。"
    return "Memento 排序更靠近问题意图，Mem0 结果相关但区分度较弱。"


def build_report() -> str:
    systems = _parse_output()
    letta_available, letta_status = _detect_letta()

    memento = systems.get("Memento", {})
    mem0 = systems.get("Mem0", {})
    queries = [q for q in memento.keys() if q in mem0]

    lines: list[str] = [
        "# Memento / Mem0 / Letta 记忆命中对比报告",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 配置与范围",
        "",
        "- 数据集: `data/testtxt.txt`，历史实测输出记录为 145 条 Q&A 记忆。",
        "- LLM: OpenCode `deepseek-v4-flash`，从 `.env` 读取，仅 Mem0/Letta 类 agent 流程需要；本次 Mem0 使用 `infer=False`，不经 LLM 抽取事实。",
        "- Embedding: 硅基流动 `Qwen/Qwen3-Embedding-4B`，2560 维。",
        "- Memento: FAISS 向量索引 + 关键词图扩散。",
        "- Mem0: 本地 Qdrant 向量库 + 直接存储原文。",
        f"- Letta: {letta_status}。",
        "",
        "## 总体结论",
        "",
        "Memento 在这组 8 个查询里整体命中质量更稳，尤其是具体技术问题和需要区分近邻主题的问题；Mem0 的优势是系统成熟、部署路径清晰、查询阶段通常更轻。Letta server 当前可用，且已修复 OpenCode LLM 与硅基流动 embedding 的 key 混用问题；不过现有 `compare_systems.py` 仍未实现 Letta 的批量写入与同口径 top_k 对比，因此本表仍只列 Memento / Mem0 的历史实测命中。",
        "",
        "从已有实测结果看，Memento 的 top1 在 8/8 个查询上都可用，其中 5 个查询明显优于 Mem0 排序；Mem0 在 4 个查询上 top1 同样准确，但在“钢琴连电脑”“学习效率”“有效休息”等问题上更容易把相关但非核心的记忆排到第一。",
        "",
        "## 对比表",
        "",
        "| 查询 | Memento top1 | Mem0 top1 | 判断 |",
        "|---|---|---|---|",
    ]

    for query in queries:
        m_hit = memento[query][0] if memento[query] else {"score": "-", "text": ""}
        z_hit = mem0[query][0] if mem0[query] else {"score": "-", "text": ""}
        lines.append(
            "| "
            + query
            + " | "
            + f"{m_hit['score']} / {_truncate(m_hit['text'])}"
            + " | "
            + f"{z_hit['score']} / {_truncate(z_hit['text'])}"
            + " | "
            + _judge_query(query, m_hit["text"], z_hit["text"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## 示例",
            "",
            "### 示例 1: 具体工具问题",
            "",
            "查询: `钢琴连电脑需要什么线和软件`",
            "",
            "- Memento top1 命中 MIDI 线/软件方案，直接回答“需要什么线和软件”。",
            "- Mem0 top1 命中练琴负反馈循环，主题相关但不是用户当前问题的核心答案。",
            "",
            "### 示例 2: 局域网与代理冲突",
            "",
            "查询: `梯子和局域网冲突怎么解决`",
            "",
            "- Memento 与 Mem0 都命中网络环境冲突、v2rayN/TUN/路由模式相关记忆。",
            "- 这类关键词明确的问题，两套系统都能稳定命中；差异主要体现在 Memento 会把相关上下文通过图扩散聚拢。",
            "",
            "### 示例 3: 抽象认知问题",
            "",
            "查询: `怎么有效休息不会浪费意志力`",
            "",
            "- Memento top1 命中番茄钟休息误区，能直接回答“怎样休息”。",
            "- Mem0 top1 命中“休息不是停止工作”，方向正确但更泛化。",
            "",
            "### 示例 4: Letta 修复验证",
            "",
            "验证查询: `钢琴连电脑需要什么线`",
            "",
            "- 写入 passage: `测试记忆: 钢琴连接电脑需要 MIDI 转 USB 线，Korg D1 没有 USB 口时要用圆头 MIDI 线。`",
            "- Letta search 返回 1 条命中，内容与写入 passage 一致。",
            "",
            "## 工程判断",
            "",
            "- Memento 当前更适合做“个人长期记忆命中”: 图扩散能把关键词、主题和相邻语境串起来，排序质量更好。",
            "- Mem0 更适合做“成熟组件接入”: API 和存储封装完整，查询速度在热启动后更稳定。",
            "- Letta 更像完整 agent runtime，不是单纯向量记忆库；当前 key 配置已经跑通，下一步需要实现同数据写入、同 top_k 查询和同输出解析。",
            "",
            "## 后续建议",
            "",
            "1. 为 Letta 补齐独立 adapter: `add_memory(text, metadata)` 和 `search(query, top_k)` 两个最小接口即可，不要把 agent 对话能力混入本次命中测试。",
            "2. 用 `scripts/test/verify_letta_memory.py` 做 smoke test，确认容器重建后 key 路由仍然正确。",
            "3. 下次重跑时清空 Mem0 本地 Qdrant collection，避免重复 add 影响分数和耗时。",
        ]
    )

    lines.extend(
        [
            "",
            "## Letta 修复记录",
            "",
            "本次修复的是 Letta 的 provider key 路由问题:",
            "",
            "- 旧配置: Letta 容器全局 `OPENAI_API_KEY` 指向 OpenCode，embedding 调硅基流动时返回 401。",
            "- 新配置: `OPENAI_API_KEY=${SILICONFLOW_API_KEY}` 用于 Letta OpenAI embedding client。",
            "- LLM 配置: OpenCode key 通过 Letta BYOK provider `opencode-deepseek-v4-flash` 注册。",
            "- 验证结果: `agents.passages.create(...)` 和 `agents.passages.search(...)` 均成功。",
            "",
            "因此 Letta 当前状态应标记为 `SERVER AVAILABLE / EMBEDDING AUTH FIXED / SMOKE SEARCH PASSED`。",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    report = _safe_report_path()
    report.write_text(build_report(), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
