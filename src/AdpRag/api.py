# src/AdpRag/api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from time import time

from AdpRag.vector_store import RAGVectorStore
from AdpRag.qa import create_qa_chain
from AdpRag.reranker import RAGReranker
from AdpRag.logger import FileLogger as log
from AdpRag.config import CHROMA_DIR, TOP_K, RERANKER_TOP_K

# ── FastAPI initialization ────────────────────────────────────────────────
app = FastAPI(
    title="Internal Docs Q&A API",
    description="RAG system for answering questions from internal company documents.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceInfo(BaseModel):
    document: str
    chunk_preview: str
    relevance_score: float | None = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    found_in_docs: bool
    sources: list[SourceInfo]
    steps: list[str]
    duration_seconds: float


# ── Init ─────────────────────────────────────────────────────────────────
if not Path(CHROMA_DIR).exists():
    raise RuntimeError(f"ChromaDB does not exist ({CHROMA_DIR}). Run setup first!")

rag_store = RAGVectorStore(chroma_dir=CHROMA_DIR)
vectorstore = rag_store.load_vectorstore()

qa_chain = create_qa_chain()
reranker = RAGReranker.get()

log.info("API initialized successfully. Ready to serve questions.")


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Internal Docs Q&A API",
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_in_db": vectorstore._collection.count(),
        "chroma_dir": str(CHROMA_DIR),
    }


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    start_time = time()
    steps = []

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    steps.append(f"Received question: \"{question}\"")

    # ── Step 1: Retrieval ─────────────────────────────────────
    initial_k = TOP_K
    steps.append(f"Retrieving top {initial_k} chunks from vector DB...")

    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=initial_k
    )

    if not docs_with_scores:
        steps.append("No relevant chunks found.")
        return QuestionResponse(
            question=question,
            answer="This information is not available in the internal documents.",
            found_in_docs=False,
            sources=[],
            steps=steps,
            duration_seconds=round(time() - start_time, 2),
        )

    # ── Step 2: Reranking ─────────────────────────────────────
    steps.append("Reranking documents...")
    try:
        docs_with_scores = reranker.rerank(
            question,
            docs_with_scores,
            top_k=RERANKER_TOP_K
        )
    except Exception as e:
        log.warning(f"Reranker failed: {e}")

    for doc, score in docs_with_scores:
        src = doc.metadata.get("source", "unknown")
        steps.append(f"  📄 {src} — final score: {score:.3f}")

    # ── Step 3: QA ────────────────────────────────────────────
    steps.append("Generating answer via LLM...")

    reranked_docs = [doc for doc, _ in docs_with_scores]

    result = qa_chain.invoke(question, reranked_docs)
    answer = result["result"].strip()
    source_docs = result["source_documents"]

    # ── Step 4: Sources ───────────────────────────────────────
    seen = set()
    sources = []
    score_map = {id(doc): score for doc, score in docs_with_scores}

    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ").strip() + "..."
        key = (src, preview[:40])

        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(
                document=src,
                chunk_preview=preview,
                relevance_score=round(score_map.get(id(doc), 0.0), 3)
            ))

    # ── Step 5: Validation ────────────────────────────────────
    not_found_phrases = [
        "not available",
        "cannot find",
        "no information",
        "not found",
    ]

    found_in_docs = not any(p in answer.lower() for p in not_found_phrases)
    steps.append(f"Answer generated. Found in documents: {found_in_docs}")

    duration = round(time() - start_time, 2)
    steps.append(f"✅ Finished in {duration}s. Used {len(sources)} sources.")

    return QuestionResponse(
        question=question,
        answer=answer,
        found_in_docs=found_in_docs,
        sources=sources,
        steps=steps,
        duration_seconds=duration,
    )