"""Diagnose numerical differences between sequential and batched RAG."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever


def load_queries(limit):
    path = ROOT / "results/retrieval_cache_100.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit]


def encode_batched(encoder, texts, batch_size):
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(
            encoder.encode_many(
                texts[start:start + batch_size],
                batch_size=batch_size,
            )
        )
    return np.concatenate(chunks, axis=0)


def retrieve_sequential(retriever, vectors, k):
    return [retriever.retrieve(v, k=k) for v in vectors]


def retrieve_batched(retriever, vectors, batch_size, k):
    results = []
    for start in range(0, len(vectors), batch_size):
        results.extend(
            retriever.retrieve_many(
                vectors[start:start + batch_size], k=k
            )
        )
    return results


def compare_results(reference, candidate):
    exact = []
    overlaps = []
    score_diffs = []

    for ref, new in zip(reference, candidate):
        ref_ids = ref["record_ids"]
        new_ids = new["record_ids"]

        exact.append(ref_ids == new_ids)
        overlaps.append(
            len(set(ref_ids) & set(new_ids)) / len(ref_ids)
        )

        ref_scores = dict(zip(ref_ids, ref["scores"]))
        new_scores = dict(zip(new_ids, new["scores"]))

        for record_id in set(ref_scores) & set(new_scores):
            score_diffs.append(
                abs(ref_scores[record_id] - new_scores[record_id])
            )

    return {
        "exact_match": np.mean(exact),
        "topk_overlap": np.mean(overlaps),
        "max_score_diff": max(score_diffs, default=0.0),
    }


def print_comparison(name, reference, candidate):
    result = compare_results(reference, candidate)
    print(f"\n{name}")
    print(f"  Exact ranking agreement: {result['exact_match']:.4f}")
    print(f"  Mean top-k overlap:      {result['topk_overlap']:.4f}")
    print(f"  Max shared-score diff:   {result['max_score_diff']:.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    if min(args.batch_size, args.limit, args.k) < 1:
        parser.error("batch-size, limit and k must be positive")

    rows = load_queries(args.limit)
    texts = [row["transcript"] for row in rows]

    print(f"Queries: {len(texts)}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Retrieval k: {args.k}")

    print("\nLoading encoder...")
    encoder = LocalEncoder(
        ROOT / "models/enc_a",
        device=args.device,
        dtype=args.dtype,
    )

    print("Loading retriever...")
    data_dir = ROOT / "data/rag_corpus"
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )

    print("\nEncoding sequentially...")
    sequential_vectors = np.stack([
        encoder.encode(text) for text in texts
    ])

    print("Encoding in batches...")
    batched_vectors = encode_batched(
        encoder, texts, args.batch_size
    )

    differences = np.abs(sequential_vectors - batched_vectors)
    cosine_similarities = np.sum(
        sequential_vectors * batched_vectors, axis=1
    )

    print("\nEMBEDDING COMPARISON")
    print(f"  Maximum absolute difference: {differences.max():.8f}")
    print(f"  Mean absolute difference:    {differences.mean():.8f}")
    print(f"  Minimum cosine similarity:    {cosine_similarities.min():.8f}")

    print("\nRetrieving sequential embeddings sequentially...")
    reference = retrieve_sequential(
        retriever, sequential_vectors, args.k
    )

    print("Retrieving batched embeddings sequentially...")
    batched_encoding_only = retrieve_sequential(
        retriever, batched_vectors, args.k
    )

    print("Retrieving sequential embeddings in batches...")
    batched_retrieval_only = retrieve_batched(
        retriever, sequential_vectors, args.batch_size, args.k
    )

    print("Retrieving batched embeddings in batches...")
    combined = retrieve_batched(
        retriever, batched_vectors, args.batch_size, args.k
    )

    print("\nRETRIEVAL COMPARISON (against sequential reference)")

    print_comparison(
        "1. Batched encoding, sequential retrieval",
        reference, batched_encoding_only
    )
    print_comparison(
        "2. Sequential encoding, batched retrieval",
        reference, batched_retrieval_only
    )
    print_comparison(
        "3. Batched encoding, batched retrieval",
        reference, combined
    )

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
