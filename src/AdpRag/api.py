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
from AdpRag.config import CHROMA_DIR, RERANKER_TOP_K
from AdpRag.agent import RAGAgent

# ── FastAPI initialization ────────────────────────────────────────────────
app = FastAPI(
    title="Internal Docs Q&A API",
    description="Agentic RAG system for answering questions from internal company documents.",
    version="2.0.0",
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


# ── Init ──────────────────────────────────────────────────────────────────
if not Path(CHROMA_DIR).exists():
    raise RuntimeError(f"ChromaDB does not exist ({CHROMA_DIR}). Run setup first!")

rag_store   = RAGVectorStore(chroma_dir=CHROMA_DIR)
vectorstore = rag_store.load_vectorstore()

qa_chain = create_qa_chain()
reranker = RAGReranker.get()
agent    = RAGAgent(vectorstore)

log.info("API v2 initialized (agentic mode). Ready to serve questions.")


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Internal Docs Q&A API (Agentic RAG)",
        "version": "2.0.0",
        "endpoints": {
            "ask":    "POST /ask",
            "health": "GET /health",
            "docs":   "GET /docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_in_db": vectorstore._collection.count(),
        "chroma_dir":   str(CHROMA_DIR),
    }


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    start_time = time()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Phase 1 & 2: Agent plans + retrieves iteratively ─────────────────
    agent_result      = agent.run(question)
    steps             = agent_result["steps"]
    docs_with_scores  = agent_result["docs_with_scores"]

    if not docs_with_scores:
        steps.append("❌ No relevant chunks found after all iterations.")
        return QuestionResponse(
            question=question,
            answer="This information is not available in the internal documents.",
            found_in_docs=False,
            sources=[],
            steps=steps,
            duration_seconds=round(time() - start_time, 2),
        )

    # ── Phase 3: Reranking ────────────────────────────────────────────────
    steps.append(f"⚖️  [Reranker] Reranking {len(docs_with_scores)} chunks → keeping top {RERANKER_TOP_K}...")
    try:
        docs_with_scores = reranker.rerank(
            question,
            docs_with_scores,
            top_k=RERANKER_TOP_K,
        )
        for doc, score in docs_with_scores:
            src = doc.metadata.get("source", "unknown")
            steps.append(f"   📄 {src} — final score: {score:.3f}")
    except Exception as e:
        log.warning(f"Reranker failed: {e}")
        steps.append(f"   ⚠️  Reranker failed ({e}), using unranked docs")

    # ── Phase 4: Answer generation ────────────────────────────────────────
    steps.append("💬 [LLM] Generating answer from collected context...")
    reranked_docs = [doc for doc, _ in docs_with_scores]
    result        = qa_chain.invoke(question, reranked_docs)
    answer        = result["result"].strip()
    source_docs   = result["source_documents"]

    # ── Phase 5: Build sources list ───────────────────────────────────────
    seen      = set()
    sources   = []
    score_map = {id(doc): score for doc, score in docs_with_scores}

    for doc in source_docs:
        src     = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ").strip() + "..."
        key     = (src, preview[:40])

        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(
                document=src,
                chunk_preview=preview,
                relevance_score=round(score_map.get(id(doc), 0.0), 3),
            ))

    # ── Phase 6: Validate answer ──────────────────────────────────────────
    not_found_phrases = [
        "not available",
        "cannot find",
        "no information",
        "not found",
    ]
    found_in_docs = not any(p in answer.lower() for p in not_found_phrases)

    duration = round(time() - start_time, 2)
    steps.append(f"✅ [Done] Answer generated. Found in docs: {found_in_docs}. Used {len(sources)} sources.")
    steps.append(f"⏱️  Total duration: {duration}s")

    return QuestionResponse(
        question=question,
        answer=answer,
        found_in_docs=found_in_docs,
        sources=sources,
        steps=steps,
        duration_seconds=duration,
    )