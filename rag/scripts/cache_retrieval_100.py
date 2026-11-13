from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever


def main():
    data_dir = ROOT / "data" / "rag_corpus"
    cache_path = ROOT / ".cache" / "practice_transcripts.jsonl"
    output_path = ROOT / "results" / "retrieval_cache_100.jsonl"

    data = CourseData(data_dir)
    items = data.audio_items("practice")[:100]

    transcripts = {}
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                transcripts[row["query_id"]] = row["transcript"]

    missing = [
        item["query_id"]
        for item in items
        if item["query_id"] not in transcripts
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} transcripts; "
            f"first missing query: {missing[0]}"
        )

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. "
            "Rename it before rerunning."
        )

    print("Loading encoder...", flush=True)
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device="cpu")

    print("Loading retriever...", flush=True)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings" / "corpus_enc_a.npy",
        data_dir / "embeddings" / "corpus_enc_a_ids.json",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(items, start=1):
            qid = item["query_id"]
            transcript = transcripts[qid]

            query_vector = encoder.encode(transcript)
            retrieval = retriever.retrieve(query_vector, k=20)

            row = {
                "query_id": qid,
                "split": "practice",
                "transcript": transcript,
                "retrieved_record_ids": retrieval["record_ids"],
                "retrieved_scores": retrieval["scores"],
                "retrieval_ms": retrieval["latency_ms"],
            }

            out.write(json.dumps(row) + "\n")
            out.flush()

            print(
                f"[{i:3d}/{len(items)}] {qid} "
                f"top_score={retrieval['scores'][0]:.6f}",
                flush=True,
            )

    print(f"\nSaved {len(items)} questions to {output_path}")


if __name__ == "__main__":
    main()
