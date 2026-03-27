# src/AdpRag/reranker.py

from sentence_transformers import CrossEncoder
from .config import MIN_RELEVANCE
from .logger import FileLogger as log
from .config import QUALITY_SCORE_WEIGHT, CROSS_ENCODER_VALUE_WEIGHT

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
        pairs     = [(query, doc.page_content) for doc in docs]
        ce_scores = self.model.predict(pairs)

        # ── Combine with quality scores from metadata ─────────────────────
        results = []
        for doc, ce_score in zip(docs, ce_scores):
            quality_score = float(doc.metadata.get("quality_score", 0.5))
            final_score   = (ce_score * CROSS_ENCODER_VALUE_WEIGHT) + (quality_score * QUALITY_SCORE_WEIGHT )

            log.info(
                f"  [{doc.metadata.get('source', '?')}] "
                f"ce={ce_score:.3f} quality={quality_score:.2f} → final={final_score:.3f}"
            )

            if final_score >= MIN_RELEVANCE:
                results.append((doc, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]