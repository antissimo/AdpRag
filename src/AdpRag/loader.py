# src/AdpRag/loader.py

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker

from .config import DOCUMENTS_DIR, EMBED_MODEL, CHUNKING_THRESHOLD,QUALITY_DROP_THRESHOLD
from .embedder import RAGEmbedder
from .logger import FileLogger as log
from .instructions import CHUNK_QUALITY_PROMPT


# Chunks with quality_score below this threshold are dropped at ingestion time


class RAGLoader:
    def __init__(self, documents_dir=DOCUMENTS_DIR):
        self.documents_dir = Path(documents_dir)
        self.embed_model = EMBED_MODEL
        self.embeddings = RAGEmbedder.get()

    # ── Public ────────────────────────────────────────────────────────────

    def load_documents(self):
        if not self.documents_dir.exists():
            log.error(f"Folder '{self.documents_dir}' does not exist!")
            return []

        md_files = list(self.documents_dir.glob("**/*.md"))
        if not md_files:
            log.warning(f"There are no .md files in '{self.documents_dir}'!")
            return []

        # ── Load raw documents ────────────────────────────────────────────
        documents = DirectoryLoader(
            str(self.documents_dir),
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
        ).load()

        for doc in documents:
            doc.metadata["source"] = Path(doc.metadata.get("source", "")).name

        log.info(f"Loaded {len(documents)} pages/sections")

        # ── Semantic chunking ─────────────────────────────────────────────
        splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=CHUNKING_THRESHOLD,
        )
        chunks = splitter.split_documents(documents)
        log.info(f"Created {len(chunks)} chunks — starting quality scoring...")

        # ── Parallel quality scoring ──────────────────────────────────────
        chunks = self._score_chunks_parallel(chunks)

        # ── Drop low quality chunks ───────────────────────────────────────
        before = len(chunks)
        chunks = [c for c in chunks if c.metadata.get("quality_score", 1.0) >= QUALITY_DROP_THRESHOLD]
        dropped = before - len(chunks)
        log.info(f"Quality filter: dropped {dropped} low-quality chunks, keeping {len(chunks)}")

        return chunks

    # ── Private ───────────────────────────────────────────────────────────

    def _score_chunks_parallel(self, chunks: list, max_workers: int = 6) -> list:
        """
        Score all chunks in parallel using ThreadPoolExecutor.
        Each chunk gets quality_score, content_score, doc_score, quality_reason
        written into its metadata.
        """
        total = len(chunks)
        scored = [None] * total  # preserve order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._score_single_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scored[idx] = future.result()
                except Exception as e:
                    log.warning(f"Quality scoring failed for chunk {idx}: {e}. Using fallback.")
                    chunk = chunks[idx]
                    chunk.metadata["quality_score"]   = 0.5
                    chunk.metadata["content_score"]   = 0.5
                    chunk.metadata["doc_score"]       = 0.5
                    chunk.metadata["quality_reason"]  = "scoring failed — fallback"
                    scored[idx] = chunk

                completed += 1
                if completed % 10 == 0 or completed == total:
                    log.info(f"  Quality scoring: {completed}/{total} chunks done")

        return scored

    def _score_single_chunk(self, chunk):
        """Score one chunk via LLM and write results into its metadata."""
        # Lazy import to avoid circular dependency
        from .llm import RAGLLM
        llm = RAGLLM.get()

        filename = chunk.metadata.get("source", "unknown")
        # Truncate content to keep prompt manageable
        content  = chunk.page_content[:600]

        prompt = CHUNK_QUALITY_PROMPT.format(
            filename=filename,
            content=content,
        )

        response = llm.invoke(prompt)
        parsed   = self._parse_quality_response(response)

        chunk.metadata["quality_score"]  = parsed["quality_score"]
        chunk.metadata["content_score"]  = parsed["content_score"]
        chunk.metadata["doc_score"]      = parsed["doc_score"]
        chunk.metadata["quality_reason"] = parsed["reason"]

        log.info(
            f"  [{filename}] quality={parsed['quality_score']:.2f} "
            f"(content={parsed['content_score']:.2f}, doc={parsed['doc_score']:.2f}) "
            f"— {parsed['reason']}"
        )
        return chunk

    def _parse_quality_response(self, response: str) -> dict:
        """Parse LLM JSON response, with safe fallback."""
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in response")

            data = json.loads(match.group(0))

            return {
                "quality_score": float(data.get("quality_score", 0.5)),
                "content_score": float(data.get("content_score", 0.5)),
                "doc_score":     float(data.get("doc_score", 0.5)),
                "reason":        str(data.get("reason", "")),
            }
        except Exception as e:
            log.warning(f"Failed to parse quality response: {e}. Raw: {response!r}")
            return {
                "quality_score": 0.5,
                "content_score": 0.5,
                "doc_score":     0.5,
                "reason":        "parse failed — fallback",
            }