"""KeyAtten 锚定词 → 预测缺失词 → 按覆盖率召回整句原文的最小闭环。"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memento.index.keyatten_extractor import MemoryKeywordExtractor


@dataclass(frozen=True)
class SentenceMemory:
    id: str
    text: str
    anchors: tuple[str, ...]


class AnchorSentenceMemory:
    """用锚定词记录句子，并以词共现做一跳预测。"""

    def __init__(self, extractor: MemoryKeywordExtractor) -> None:
        self._extractor = extractor
        self._sentences: list[SentenceMemory] = []
        self._anchor_to_sentence_ids: dict[str, set[str]] = defaultdict(set)
        self._association: dict[str, Counter[str]] = defaultdict(Counter)

    def remember(self, sentence_id: str, text: str) -> SentenceMemory:
        anchors = tuple(self._extractor.extract(text, top_k=10))
        memory = SentenceMemory(sentence_id, text, anchors)
        self._sentences.append(memory)

        for anchor in anchors:
            self._anchor_to_sentence_ids[anchor].add(sentence_id)
        for source in anchors:
            for target in anchors:
                if source != target:
                    self._association[source][target] += 1
        return memory

    def recall(
        self,
        query: str,
        *,
        min_prediction_support: int = 2,
        min_prediction_ratio: float = 0.60,
        prediction_weight: float = 0.55,
        recall_threshold: float = 0.75,
    ) -> tuple[list[str], dict[str, float], list[dict[str, object]]]:
        direct_anchors = self._extractor.extract(query, top_k=10)
        direct_set = set(direct_anchors)
        predicted = self._predict(
            direct_set,
            min_support=min_prediction_support,
            min_ratio=min_prediction_ratio,
        )

        candidate_ids: set[str] = set()
        for anchor in direct_set | set(predicted):
            candidate_ids.update(self._anchor_to_sentence_ids.get(anchor, set()))

        results = []
        for memory in self._sentences:
            if memory.id not in candidate_ids or not memory.anchors:
                continue
            direct = [anchor for anchor in memory.anchors if anchor in direct_set]
            inferred = [anchor for anchor in memory.anchors if anchor in predicted]
            coverage = (
                len(direct)
                + sum(predicted[anchor] * prediction_weight for anchor in inferred)
            ) / len(memory.anchors)
            if coverage >= recall_threshold:
                results.append(
                    {
                        "id": memory.id,
                        "text": memory.text,
                        "score": round(coverage, 3),
                        "direct_anchors": direct,
                        "predicted_anchors": inferred,
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return direct_anchors, predicted, results

    def _predict(
        self,
        direct_anchors: set[str],
        *,
        min_support: int,
        min_ratio: float,
    ) -> dict[str, float]:
        predicted: dict[str, float] = {}
        for candidate, neighbors in self._association.items():
            if candidate in direct_anchors:
                continue
            total = sum(neighbors.values())
            support = sum(neighbors[anchor] for anchor in direct_anchors)
            ratio = support / total if total else 0.0
            if support >= min_support and ratio >= min_ratio:
                predicted[candidate] = round(ratio, 3)
        return predicted


def make_extractor() -> MemoryKeywordExtractor:
    return MemoryKeywordExtractor(
        model_path=str(ROOT / "models" / "Qwen3-Embedding-0.6B"),
        device="cuda",
        dtype="float16",
        default_top_k=10,
        phrase_merge_enabled=True,
        cache_enabled=True,
        cache_dir=ROOT / "data" / "keyatten_cache",
    )


def main() -> None:
    memory = AnchorSentenceMemory(make_extractor())
    stored = [
        (
            "delivery",
            "上周客户要求我们下周完成项目交付，但预算审批尚未通过，团队正在调整实施方案。",
        ),
        (
            "team_building",
            "行政部门确认下周组织团建，财务已经批准活动预算，运营组正在预订场地。",
        ),
        (
            "new_requirement",
            "客户今天确认新增需求，产品团队将在本周提交实施方案。",
        ),
    ]
    print("=== 写入记忆：句子与 KeyAtten 锚定词 ===")
    for sentence_id, text in stored:
        item = memory.remember(sentence_id, text)
        print(f"{item.id}: {list(item.anchors)}")

    query = "客户在问项目进度，预算审批现在有结果了吗？"
    direct, predicted, results = memory.recall(query)
    print("\n=== 查询 ===")
    print(query)
    print(f"直接锚定词: {direct}")
    print(f"预测锚定词（词: 支持比例）: {predicted}")

    print("\n=== 覆盖率达到 0.75 后召回的原句 ===")
    if not results:
        print("（无句子达到阈值）")
    for result in results:
        print(f"[{result['id']}] score={result['score']}")
        print(f"  原文: {result['text']}")
        print(f"  直接命中: {result['direct_anchors']}")
        print(f"  预测补全: {result['predicted_anchors']}")


if __name__ == "__main__":
    main()
