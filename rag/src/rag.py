
from __future__ import annotations

import time


class RAGPipeline:
    """Audio-to-answer RAG pipeline with optional adaptive evidence selection."""

    def __init__(
        self,
        asr,
        encoder,
        retriever,
        data,
        context_builder,
        generator,
        evidence_selector=None,
    ):
        self.asr = asr
        self.encoder = encoder
        self.retriever = retriever
        self.data = data
        self.context_builder = context_builder
        self.generator = generator
        self.evidence_selector = evidence_selector

    def answer_audio(self, audio_path, k: int = 5):
        total_start = time.perf_counter()

        # Step 1: Transcribe audio
        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio_path)
        asr_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: Encode transcript
        t0 = time.perf_counter()
        query_vector = self.encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        # Step 3: Retrieve top-K candidate documents
        retrieval = self.retriever.retrieve(query_vector, k=k)

        # Step 4: Optionally select evidence adaptively
        selection = None
        selected_ids = retrieval["record_ids"]

        t0 = time.perf_counter()

        if self.evidence_selector is not None:
            selection = self.evidence_selector.select(
                retrieval["record_ids"],
                retrieval["scores"],
                self.data,
            )
            selected_ids = selection["record_ids"]

        selection_ms = (time.perf_counter() - t0) * 1000.0

        # Step 5: Construct prompt using selected evidence
        built = self.context_builder.build(
            transcript,
            selected_ids,
            self.data,
        )

        # Step 6: Generate answer
        generation = self.generator.generate(built["prompt"])

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "transcript": transcript,
            "retrieval": retrieval,
            "selection": selection,
            "selected_record_ids": selected_ids,
            "contexts": built["contexts"],
            "answer": generation["answer"],
            "timing": {
                "asr_ms": asr_ms,
                "embedding_ms": embedding_ms,
                "retrieval_ms": retrieval["latency_ms"],
                "selection_ms": selection_ms,
                "generation_ms": generation["latency_ms"],
                "end_to_end_ms": total_ms,
            },
        }

