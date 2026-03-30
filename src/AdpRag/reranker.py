# src/AdpRag/reranker.py

import numpy as np
from sentence_transformers import CrossEncoder
from .config import MIN_RELEVANCE, QUALITY_SCORE_WEIGHT, CROSS_ENCODER_WEIGHT
from .logger import FileLogger as log


class RAGReranker:

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        log.info("Loading reranker model (CrossEncoder)...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        log.info("Reranker ready.")

    def rerank(self, query: str, docs_with_scores: list, top_k: int = 5) -> list:
        if not docs_with_scores:
            return []

        docs, _ = zip(*docs_with_scores)

        # ── CrossEncoder relevance scores ─────────────────────────────────
        pairs = [(query, doc.page_content) for doc in docs]
        ce_scores = self.model.predict(pairs)

        # ── Normalize CE scores to [0, 1] via sigmoid ─────────────────────
        ce_scores_norm = self._sigmoid(ce_scores)

        # ── Combine with quality scores from metadata ─────────────────────
        results = []
        for doc, ce_score, ce_norm in zip(docs, ce_scores, ce_scores_norm):
            quality_score = float(doc.metadata.get("quality_score", 0.5))
            final_score = (ce_norm * CROSS_ENCODER_WEIGHT) + (quality_score * QUALITY_SCORE_WEIGHT)

            log.info(
                f"  [{doc.metadata.get('source', '?')}] "
                f"ce_raw={ce_score:.3f} ce_norm={ce_norm:.3f} "
                f"quality={quality_score:.2f} → final={final_score:.3f}"
            )

            if final_score >= MIN_RELEVANCE:
                results.append((doc, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _sigmoid(x: np.ndarray, temperature: float = 3.0) -> np.ndarray:
        """
        Normalize scores to [0, 1] via sigmoid.
        Temperature controls the steepness:
        - Lower → more aggressive separation
        - Higher → scores cluster toward 0.5
        """
        return 1 / (1 + np.exp(-x / temperature))