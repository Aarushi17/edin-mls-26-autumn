
"""Generate RAG answers with optional adaptive evidence selection."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_selector import AdaptiveEvidenceSelector
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--split", choices=["practice", "dev"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--mode", choices=["fixed", "adaptive"], required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--gap-threshold", type=float, default=0.015)

    args = parser.parse_args()

    if args.k < 1 or args.candidate_k < 1:
        parser.error("K values must be positive")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if args.output.exists():
        parser.error("Output exists; choose a new filename")

    selector = None

    if args.mode == "adaptive":
        if args.candidate_k < args.max_k:
            parser.error("--candidate-k must be >= --max-k")

        selector = AdaptiveEvidenceSelector(
            token_budget=args.token_budget,
            min_k=args.min_k,
            max_k=args.max_k,
            gap_threshold=args.gap_threshold,
        )

    asr = ASRAdapter()
    asr.prepare()

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)

    pipeline = RAGPipeline(
        asr=asr,
        encoder=LocalEncoder(ROOT / "models/enc_a", device=args.device),
        retriever=ExactDenseRetriever(
            data_dir / "embeddings/corpus_enc_a.npy",
            data_dir / "embeddings/corpus_enc_a_ids.json",
        ),
        data=data,
        context_builder=SimpleContextBuilder(),
        generator=ELMGenerator(),
        evidence_selector=selector,
    )

    items = data.audio_items(args.split)

    if args.limit is not None:
        items = items[:args.limit]

    retrieval_k = args.k if args.mode == "fixed" else args.candidate_k

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as file:
        for item in items:
            result = pipeline.answer_audio(
                data_dir / item["audio_file"],
                k=retrieval_k,
            )

            selection = result["selection"]

            row = {
                "query_id": item["query_id"],
                "split": args.split,
                "audio_file": item["audio_file"],
                "mode": args.mode,
                "transcript": result["transcript"],
                "answer": result["answer"],
                "retrieved_record_ids": result["retrieval"]["record_ids"],
                "selected_record_ids": result["selected_record_ids"],
                "selected_k": len(result["selected_record_ids"]),
                "selection": selection,
                "timing": result["timing"],
            }

            if args.mode == "adaptive":
                row["token_budget"] = args.token_budget

            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()

            print(
                f'{item["query_id"]}: '
                f'selected_k={row["selected_k"]}'
            )


if __name__ == "__main__":
    main()

