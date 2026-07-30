"""模式补全最小验证：局部联想是否能可靠地唤起已学情境。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping


@dataclass(frozen=True)
class Pattern:
    """一次已学习情境的高阶表示，成员是可读的中文语义特征。"""

    name: str
    members: frozenset[str]
    min_coverage: float = 0.80
    min_direct_support: int = 2


def extract_features(text: str, vocabulary: frozenset[str]) -> dict[str, float]:
    """最小文本入口：识别输入中出现的特征短语。

    这里故意只用精确短语匹配，隔离验证“词语激活 → 模式补全”链路；
    正式接入时应替换为项目已有的关键词抽取和向量语义匹配。
    """
    return {feature: 1.0 for feature in vocabulary if feature in text}


def spread_once(
    direct_activations: Mapping[str, float],
    associations: Mapping[str, Mapping[str, float]],
    decay: float = 0.80,
) -> dict[str, float]:
    """仅做一跳二元联想，返回直接与联想后的最大激活值。"""
    activations = dict(direct_activations)
    for source, activation in direct_activations.items():
        for target, weight in associations.get(source, {}).items():
            inferred = activation * weight * decay
            activations[target] = max(activations.get(target, 0.0), inferred)
    return activations


def match_pattern(
    pattern: Pattern,
    direct_activations: Mapping[str, float],
    activations: Mapping[str, float],
) -> float | None:
    """以直接线索数量和全体覆盖率共同决定是否激活模式。"""
    direct_support = len(pattern.members & direct_activations.keys())
    if direct_support < pattern.min_direct_support:
        return None

    coverage = sum(activations.get(member, 0.0) for member in pattern.members)
    coverage /= len(pattern.members)
    return coverage if coverage >= pattern.min_coverage else None


def complete_pattern(
    pattern: Pattern,
    direct_activations: Mapping[str, float],
    activations: Mapping[str, float],
) -> dict[str, object]:
    """报告模式命中及相对当前输入被补全的成员，不把推断写回记忆。"""
    score = match_pattern(pattern, direct_activations, activations)
    if score is None:
        return {"matched": False, "score": None, "completed": []}

    return {
        "matched": True,
        "score": round(score, 3),
        "completed": sorted(pattern.members - direct_activations.keys()),
    }


def main() -> None:
    pattern = Pattern(
        name="项目交付情境",
        members=frozenset({"客户需求", "交付日期", "预算审批", "实施方案"}),
    )
    associations = {
        "客户反馈": {"客户需求": 0.90},
        "项目排期": {"交付日期": 0.90},
    }
    vocabulary = frozenset(set(pattern.members) | set(associations))

    print("=== 正例：从中文输入提取线索并补全项目交付情境 ===")
    text = "客户反馈要求尽快确认项目排期，预算审批和实施方案需要同步处理。"
    direct = extract_features(text, vocabulary)
    activations = spread_once(direct, associations)
    result = complete_pattern(pattern, direct, activations)
    print(f"输入文本: {text}")
    print(f"直接激活: {sorted(direct)}")
    print(f"一跳联想: {sorted(activations)}")
    print(f"模式结果: {result}")

    assert isclose(activations["客户需求"], 0.72)
    assert isclose(activations["交付日期"], 0.72)
    assert result["matched"] is True
    assert result["completed"] == ["交付日期", "客户需求"]

    print("\n=== 反例：只有两个情境成员，不得错误补全 ===")
    weak_text = "预算审批已经完成，实施方案等待评审。"
    weak_direct = extract_features(weak_text, vocabulary)
    weak_activations = spread_once(weak_direct, associations)
    weak_result = complete_pattern(pattern, weak_direct, weak_activations)
    print(f"输入文本: {weak_text}")
    print(f"直接激活: {sorted(weak_direct)}")
    print(f"模式结果: {weak_result}")

    assert weak_result == {"matched": False, "score": None, "completed": []}
    print("\n结论：模式层可区分“局部联想”与“足够证据下的整体补全”。")


if __name__ == "__main__":
    main()
