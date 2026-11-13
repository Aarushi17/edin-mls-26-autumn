from __future__ import annotations

import math
import re
from numbers import Real


class AdaptiveEvidenceSelector:
    """
    Adaptive evidence selection using similarity-score gaps and
    a configurable evidence-token budget.

    Documents are considered in retrieval-rank order.

    Selection procedure:
      1. Determine a query-dependent candidate cutoff using score gaps.
      2. Include ranked documents while the token budget permits.
      3. Stop at the first document that does not fit.
      4. If no document fits, retain the highest-ranked document and
         explicitly record that the budget was exceeded.

    Token counts are estimates, not model-tokenizer measurements.
    """

    def __init__(
        self,
        token_budget: int = 1024,
        min_k: int = 3,
        max_k: int = 20,
        gap_threshold: float = 0.015,
    ):
        if token_budget < 1:
            raise ValueError("token_budget must be positive")

        if min_k < 1:
            raise ValueError("min_k must be at least 1")

        if max_k < min_k:
            raise ValueError("max_k must be >= min_k")

        if not math.isfinite(gap_threshold) or gap_threshold < 0:
            raise ValueError(
                "gap_threshold must be finite and non-negative"
            )

        self.token_budget = token_budget
        self.min_k = min_k
        self.max_k = max_k
        self.gap_threshold = gap_threshold

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Approximate token count using words and punctuation.

        This is not the ELM model's actual token count.
        """
        return len(re.findall(r"\w+|[^\w\s]", text))

    def select(
        self,
        record_ids: list[str],
        scores: list[float],
        data,
    ) -> dict:
        if len(record_ids) != len(scores):
            raise ValueError(
                "Record IDs and scores must have equal lengths"
            )

        if not record_ids:
            raise ValueError("No retrieved documents")

        # Accept Python and NumPy numeric score types.
        try:
            scores = [float(score) for score in scores]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Scores must be numeric"
            ) from exc

        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Scores must be finite")

        # Assumes retriever outputs scores in descending order.
        if any(
            scores[i] < scores[i + 1]
            for i in range(len(scores) - 1)
        ):
            raise ValueError(
                "Scores must be sorted in descending order"
            )

        limit = min(len(record_ids), self.max_k)

        # Step 1: Find a meaningful score gap.
        # Never cut off before min_k candidates when available.
        target_k = limit
        gap_triggered = False
        triggering_gap = None

        for i in range(min(self.min_k, limit), limit):
            gap = scores[i - 1] - scores[i]

            if gap >= self.gap_threshold:
                target_k = i
                gap_triggered = True
                triggering_gap = gap
                break

        # Step 2: Select a ranked prefix within the budget.
        selected_ids = []
        selected_scores = []
        used_tokens = 0
        stopped_by_budget = False

        for rid, score in zip(
            record_ids[:target_k],
            scores[:target_k],
        ):
            document = data.record_text(rid)
            cost = self.count_tokens(document)

            if used_tokens + cost > self.token_budget:
                stopped_by_budget = True
                break

            selected_ids.append(rid)
            selected_scores.append(score)
            used_tokens += cost

        # Step 3: Avoid passing an empty evidence set.
        # This fallback can exceed the configured budget.
        forced_top1 = False

        if not selected_ids:
            rid = record_ids[0]
            selected_ids = [rid]
            selected_scores = [scores[0]]
            used_tokens = self.count_tokens(
                data.record_text(rid)
            )
            forced_top1 = True

        budget_exceeded = used_tokens > self.token_budget

        return {
            "record_ids": selected_ids,
            "scores": selected_scores,
            "selected_k": len(selected_ids),
            "evidence_tokens_estimated": used_tokens,
            "target_k": target_k,
            "budget_exhausted": stopped_by_budget,
            "budget_exceeded": budget_exceeded,
            "forced_top1": forced_top1,
            "gap_triggered": gap_triggered,
            "triggering_gap": triggering_gap,
            "token_budget": self.token_budget,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "gap_threshold": self.gap_threshold,
        }
