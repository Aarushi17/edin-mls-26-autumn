"""Compare answer generation using cached retrieval results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.context_builder import SimpleContextBuilder
from src.generator import ELMGenerator
from src.adaptive_selector import AdaptiveEvidenceSelector


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def count_tokens(text):
    return AdaptiveEvidenceSelector.count_tokens(text)


def select_budget_only(record_ids, data, budget):
    selected = []
    used = 0

    for rid in record_ids:
        cost = count_tokens(data.record_text(rid))

        if used + cost > budget:
            break

        selected.append(rid)
        used += cost

    # Same fallback as the offline comparison.
    return selected or record_ids[:1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fixed_k5", "budget_3072"],
        required=True,
    )
    parser.add_argument("--start", type=int, default=50)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.start < 50 or args.start >= 100:
        parser.error("--start must be between 50 and 99")

    if args.limit < 1 or args.start + args.limit > 100:
        parser.error("Invalid limit for held-out questions")

    cache_path = ROOT / "results/retrieval_cache_100.jsonl"
    output_path = (
        ROOT / "results" /
        f"generation_{args.mode}_heldout_{args.start}_{args.limit}.jsonl"
    )

    rows = load_jsonl(cache_path)
    if len(rows) != 100:
        raise RuntimeError("Expected 100 cached retrieval rows")

    subset = rows[args.start:args.start + args.limit]
    expected_qids = [r["query_id"] for r in subset]

    data = CourseData(ROOT / "data/rag_corpus")
    builder = SimpleContextBuilder()

    # Read existing output to allow safe resumption.
    completed = {}

    if output_path.exists():
        for row in load_jsonl(output_path):
            qid = row["query_id"]
            if qid not in expected_qids:
                raise RuntimeError(
                    f"Unexpected query ID in output: {qid}"
                )
            if qid in completed:
                raise RuntimeError(
                    f"Duplicate query ID in output: {qid}"
                )
            completed[qid] = row

    print(
        f"Mode: {args.mode}, "
        f"questions: {len(subset)}, "
        f"already completed: {len(completed)}",
        flush=True,
    )

    if len(completed) == len(subset):
        print("All answers already generated.")
        return

    generator = ELMGenerator()
    print(f"Generator model: {generator.model}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as out:
        for i, row in enumerate(subset, start=1):
            qid = row["query_id"]

            if qid in completed:
                print(f"[{i}/{len(subset)}] Skipping {qid}")
                continue

            ids = row["retrieved_record_ids"]

            if args.mode == "fixed_k5":
                selected = ids[:5]
            else:
                selected = select_budget_only(
                    ids, data, budget=3072
                )

            evidence_tokens = sum(
                count_tokens(data.record_text(rid))
                for rid in selected
            )

            built = builder.build(
                row["transcript"],
                selected,
                data,
            )

            print(
                f"[{i}/{len(subset)}] {qid} "
                f"K={len(selected)} "
                f"evidence_tokens={evidence_tokens}",
                flush=True,
            )

            # Generation is the only remote model call.
            generated = generator.generate(built["prompt"])

            result = {
                "query_id": qid,
                "split": "practice",
                "mode": args.mode,
                "transcript": row["transcript"],
                "answer": generated["answer"],
                "retrieved_record_ids": ids,
                "selected_record_ids": selected,
                "selected_k": len(selected),
                "evidence_tokens_estimated": evidence_tokens,
                "prompt_tokens_estimated": count_tokens(
                    built["prompt"]
                ),
                "timing": {
                    "generation_ms": generated["latency_ms"]
                },
                "generator_model": generator.model,
            }

            out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            out.flush()

            print(
                f"  Generation: "
                f"{generated['latency_ms'] / 1000:.2f}s",
                flush=True,
            )

    print(f"\nSaved answers to: {output_path}")


if __name__ == "__main__":
    main()
