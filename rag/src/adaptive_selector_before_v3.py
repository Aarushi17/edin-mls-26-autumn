from __future__ import annotations

import math
import re


class AdaptiveEvidenceSelector:
    """Select a ranked prefix of documents within an evidence budget."""

    def __init__(
        self,
        token_budget: int = 1024,
        min_k: int = 1,
        max_k: int = 20,
        gap_threshold: float = 0.015,
    ):
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        if min_k < 1 or max_k < min_k:
            raise ValueError("Invalid min_k or max_k")
        if gap_threshold < 0:
            raise ValueError("gap_threshold must be non-negative")

        self.token_budget = token_budget
        self.min_k = min_k
        self.max_k = max_k
        self.gap_threshold = gap_threshold

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))

    def select(self, record_ids: list[str], scores: list[float], data) -> dict:
        if len(record_ids) != len(scores):
            raise ValueError("Record IDs and scores must have equal lengths")
        if not record_ids:
            raise ValueError("No retrieved documents")
        if any(
            not isinstance(s, (int, float)) or not math.isfinite(s)
            for s in scores
        ):
            raise ValueError("Scores must be finite numeric values")

        limit = min(len(record_ids), self.max_k)

        selected_ids = []
        selected_scores = []
        used_tokens = 0

        for rid, score in zip(record_ids[:limit], scores[:limit]):
            cost = self.count_tokens(data.record_text(rid))

            if used_tokens + cost > self.token_budget:
                break

            selected_ids.append(rid)
            selected_scores.append(score)
            used_tokens += cost

        return {
            "record_ids": selected_ids,
            "scores": selected_scores,
            "selected_k": len(selected_ids),
            "evidence_tokens_estimated": used_tokens,
            "target_k": limit,
            "budget_exhausted": len(selected_ids) == 0,
        }
