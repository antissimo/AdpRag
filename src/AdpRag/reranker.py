import json
from pathlib import Path
from sentence_transformers import CrossEncoder
from .config import CACHE_DIR
from .logger import FileLogger as log
from .llm import RAGLLM
from .instructions import format_reranking_prompt
from .helpers import parse_llm_json

PRIORITY_CACHE_FILE = Path(CACHE_DIR) / "priority_cache.json"

class RAGReranker:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        log.info("Loading reranker model...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = RAGLLM.get()
        self.priority_cache = self._load_cache()
        log.info(f"Reranker ready. Loaded {len(self.priority_cache)} cached priorities.")

    def rerank(self, query: str, docs_with_scores: list, top_k: int = 5) -> list:
        if not docs_with_scores:
            return []

        docs, _ = zip(*docs_with_scores)

        pairs = [(query, doc.page_content) for doc in docs]
        ce_scores = self.model.predict(pairs)

        results = []
        for doc, ce_score in zip(docs, ce_scores):
            source = doc.metadata.get("source", "unknown")
            priority = self._get_priority_llm(source, doc.page_content)
            final_score = ce_score + (priority * 2)
            results.append((doc, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def invalidate(self, source: str):
        if source in self.priority_cache:
            del self.priority_cache[source]
            self._save_cache()
            log.info(f"Cache invalidated for '{source}'")

    def clear_cache(self):
        self.priority_cache = {}
        self._save_cache()
        log.info("Priority cache cleared")

    def _get_priority_llm(self, source: str, content: str) -> float:
        log.info(f"Trying to get priority of '{source}'")
        if source in self.priority_cache:
            return self.priority_cache[source]["priority"]

        prompt = format_reranking_prompt(source)        
        log.info(f"Invoking LLM for priority of '{source}'")
        response = self.client.invoke(prompt)

        try:
            parsed = parse_llm_json(response)
            priority = float(parsed.get("priority", 0.5))
            reason = parsed.get("reason", "")
            log.info(f"Parsed JSON for '{source}': {parsed!r}")
        except Exception as e:
            log.warning(
                f"Parsing failed for '{source}': {e}. Using fallback 0.5. Raw LLM response: {response!r}"
            )
            priority = 0.5
            reason = "fallback"

        self.priority_cache[source] = {"priority": priority, "reason": reason}
        self._save_cache()
        log.info(f"Priority for '{source}': {priority} — {reason}")
        return priority

    def _load_cache(self) -> dict:
        if PRIORITY_CACHE_FILE.exists():
            try:
                with open(PRIORITY_CACHE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load priority cache: {e}")
        return {}

    def _save_cache(self):
        try:
            PRIORITY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PRIORITY_CACHE_FILE, "w") as f:
                json.dump(self.priority_cache, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save priority cache: {e}")