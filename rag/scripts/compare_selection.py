from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.adaptive_selector import AdaptiveEvidenceSelector

DATA_DIR = ROOT / "data" / "rag_corpus"
INPUT = ROOT / "results" / "retrieval_cache_100.jsonl"
OUTPUT = ROOT / "results" / "selection_comparison.csv"

data = CourseData(DATA_DIR)
qrels = data.load_qrels("practice")

with INPUT.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

assert len(rows) == 100, f"Expected 100 queries, got {len(rows)}"
assert len({r["query_id"] for r in rows}) == 100

# Cache token counts so document text is not repeatedly processed.
token_cache = {}

def tokens(rid):
    if rid not in token_cache:
        token_cache[rid] = AdaptiveEvidenceSelector.count_tokens(
            data.record_text(rid)
        )
    return token_cache[rid]

def fixed(ids, k):
    return ids[:k]

def budget_only(ids, budget):
    selected = []
    used = 0

    for rid in ids:
        cost = tokens(rid)
        if used + cost > budget:
            break
        selected.append(rid)
        used += cost

    # Match V3's top-1 fallback.
    return selected or ids[:1]

def gap_budget(ids, scores, budget, threshold):
    selector = AdaptiveEvidenceSelector(
        token_budget=budget,
        min_k=3,
        max_k=20,
        gap_threshold=threshold,
    )
    return selector.select(ids, scores, data)["record_ids"]

methods = {}

for k in [3, 5, 10, 20]:
    methods[f"fixed_k{k}"] = (
        lambda ids, scores, k=k: fixed(ids, k)
    )

for budget in [512, 768, 1024, 1536, 2048, 3072]:
    methods[f"budget_{budget}"] = (
        lambda ids, scores, budget=budget:
        budget_only(ids, budget)
    )

for budget in [1024, 1536, 2048, 3072]:
    for threshold in [0.001, 0.003, 0.005, 0.01, 0.015]:
        name = f"gap_{threshold}_budget_{budget}"
        methods[name] = (
            lambda ids, scores, budget=budget, threshold=threshold:
            gap_budget(ids, scores, budget, threshold)
        )

results = []

# First 50: tuning; last 50: held-out.
for split_name, subset in [
    ("tuning", rows[:50]),
    ("heldout", rows[50:]),
]:
    for method_name, method in methods.items():
        coverages = []
        hits = []
        token_counts = []
        selected_counts = []

        for row in subset:
            qid = row["query_id"]
            ids = row["retrieved_record_ids"]
            scores = row["retrieved_scores"]

            selected = method(ids, scores)

            # Qrels are used ONLY to evaluate, never to select.
            relevant = {
                rid for rid, grade in qrels[qid].items()
                if float(grade) > 0
            }

            retained = len(set(selected) & relevant)
            coverage = (
                retained / len(relevant)
                if relevant else 0.0
            )

            coverages.append(coverage)
            hits.append(float(retained > 0))
            token_counts.append(sum(tokens(r) for r in selected))
            selected_counts.append(len(selected))

        n = len(subset)

        results.append({
            "split": split_name,
            "method": method_name,
            "coverage": sum(coverages) / n,
            "hit_rate": sum(hits) / n,
            "mean_tokens": sum(token_counts) / n,
            "mean_k": sum(selected_counts) / n,
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "split", "method", "coverage",
        "hit_rate", "mean_tokens", "mean_k"
    ])
    writer.writeheader()
    writer.writerows(results)

print("\nTUNING RESULTS")
print(f"{'Method':<32} {'Coverage':>10} {'Hit rate':>10} {'Tokens':>10} {'Mean K':>8}")
print("-" * 76)

for r in results:
    if r["split"] == "tuning":
        print(
            f"{r['method']:<32} "
            f"{r['coverage']:>10.3f} "
            f"{r['hit_rate']:>10.3f} "
            f"{r['mean_tokens']:>10.1f} "
            f"{r['mean_k']:>8.2f}"
        )

print(f"\nSaved complete tuning and held-out results to: {OUTPUT}")
print("Held-out results are saved but not used for method selection.")
