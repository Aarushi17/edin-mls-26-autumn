"""Benchmark sequential versus batched encoding and exact retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever


def load_cache(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise RuntimeError("Retrieval cache is empty")

    qids = [row["query_id"] for row in rows]
    if len(qids) != len(set(qids)):
        raise RuntimeError("Duplicate query IDs in retrieval cache")

    return rows


def encode_in_batches(encoder, texts, batch_size):
    vectors = []
    start = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        vectors.append(
            encoder.encode_many(chunk, batch_size=batch_size)
        )

    elapsed = time.perf_counter() - start
    return np.concatenate(vectors, axis=0), elapsed


def retrieve_in_batches(retriever, vectors, batch_size, k):
    results = []
    start = time.perf_counter()

    for i in range(0, len(vectors), batch_size):
        chunk = vectors[i:i + batch_size]

        if batch_size == 1:
            results.append(retriever.retrieve(chunk[0], k=k))
        else:
            results.extend(retriever.retrieve_many(chunk, k=k))

    elapsed = time.perf_counter() - start
    return results, elapsed


def compare_results(reference, candidate):
    exact_matches = 0
    overlaps = []
    score_differences = []

    for ref, new in zip(reference, candidate):
        ref_ids = ref["record_ids"]
        new_ids = new["record_ids"]

        if ref_ids == new_ids:
            exact_matches += 1

        overlaps.append(
            len(set(ref_ids) & set(new_ids)) / len(ref_ids)
        )

        ref_scores = dict(zip(ref_ids, ref["scores"]))
        new_scores = dict(zip(new_ids, new["scores"]))

        for rid in set(ref_scores) & set(new_scores):
            score_differences.append(
                abs(ref_scores[rid] - new_scores[rid])
            )

    return {
        "exact_match_rate": exact_matches / len(reference),
        "mean_topk_overlap": float(np.mean(overlaps)),
        "max_score_difference": (
            max(score_differences) if score_differences else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if args.repeats < 1 or args.k < 1 or args.limit < 1:
        parser.error("repeats, k and limit must be positive")

    cache = load_cache(
        ROOT / "results/retrieval_cache_100.jsonl"
    )[:args.limit]

    texts = [row["transcript"] for row in cache]
    sizes = [1, 4, 8, 16, 32]

    data_dir = ROOT / "data/rag_corpus"

    print(f"Queries: {len(texts)}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Precision: {args.dtype}", flush=True)
    print(f"Repeats: {args.repeats}", flush=True)

    print("\nLoading encoder...", flush=True)
    encoder = LocalEncoder(
        ROOT / "models/enc_a",
        device=args.device,
        dtype=args.dtype,
    )

    print("Loading exact retriever...", flush=True)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )

    # Warm up the encoder and retriever.
    print("Warming up...", flush=True)
    warm_vectors = encoder.encode_many(
        texts[:min(4, len(texts))],
        batch_size=4,
    )
    retriever.retrieve(warm_vectors[0], k=args.k)
    retriever.retrieve_many(warm_vectors, k=args.k)

    # Use the original single-query methods as the reference.
    print("Computing sequential reference...", flush=True)
    reference_vectors = np.stack([
        encoder.encode(text) for text in texts
    ])

    reference_results = [
        retriever.retrieve(vec, k=args.k)
        for vec in reference_vectors
    ]

    output_rows = []

    for batch_size in sizes:
        print(f"\nBATCH SIZE {batch_size}", flush=True)

        for repeat in range(1, args.repeats + 1):
            vectors, encode_seconds = encode_in_batches(
                encoder, texts, batch_size
            )

            results, retrieval_seconds = retrieve_in_batches(
                retriever, vectors, batch_size, args.k
            )

            quality = compare_results(reference_results, results)

            embedding_difference = float(
                np.max(np.abs(vectors - reference_vectors))
            )

            total_seconds = encode_seconds + retrieval_seconds

            result = {
                "device": args.device,
                "batch_size": batch_size,
                "repeat": repeat,
                "num_queries": len(texts),
                "k": args.k,
                "encoding_seconds": encode_seconds,
                "retrieval_seconds": retrieval_seconds,
                "total_seconds": total_seconds,
                "throughput_qps": len(texts) / total_seconds,
                "mean_time_per_query_ms": (
                    total_seconds * 1000 / len(texts)
                ),
                "exact_match_rate": quality["exact_match_rate"],
                "mean_topk_overlap": quality["mean_topk_overlap"],
                "max_score_difference": quality["max_score_difference"],
                "max_embedding_difference": embedding_difference,
            }

            output_rows.append(result)

            print(
                f"Repeat {repeat}: "
                f"total={total_seconds:.3f}s, "
                f"throughput={result['throughput_qps']:.2f} q/s, "
                f"overlap={quality['mean_topk_overlap']:.4f}, "
                f"exact={quality['exact_match_rate']:.4f}",
                flush=True,
            )

    output = ROOT / f"results/batching_benchmark_{args.dtype}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(output_rows[0]),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nSUMMARY (median across repeats)")
    print(
        f"{'Batch':>6} {'Encode(s)':>11} "
        f"{'Retrieve(s)':>12} {'Total(s)':>10} "
        f"{'Q/s':>9} {'Overlap':>9}"
    )

    for size in sizes:
        group = [
            r for r in output_rows
            if r["batch_size"] == size
        ]

        def median(key):
            return float(np.median([r[key] for r in group]))

        print(
            f"{size:>6} "
            f"{median('encoding_seconds'):>11.3f} "
            f"{median('retrieval_seconds'):>12.3f} "
            f"{median('total_seconds'):>10.3f} "
            f"{median('throughput_qps'):>9.2f} "
            f"{median('mean_topk_overlap'):>9.4f}"
        )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
