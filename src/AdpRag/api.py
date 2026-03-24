# src/AdpRag/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from time import time

from AdpRag.vector_store import RAGVectorStore
from AdpRag.qa import create_qa_chain
from AdpRag.logger import FileLogger as log
from AdpRag.config import CHROMA_DIR, TOP_K, MIN_RELEVANCE

# ── FastAPI initialization ────────────────────────────────────────────────
app = FastAPI(
    title="Internal Docs Q&A API",
    description="RAG system for answering questions from internal company documents.",
    version="1.0.0",
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

if not Path(CHROMA_DIR).exists():
    raise RuntimeError(f"ChromaDB does not exist ({CHROMA_DIR}). Run setup first!")

rag_store = RAGVectorStore(chroma_dir=CHROMA_DIR)
vectorstore = rag_store.load_vectorstore()  

qa_chain = create_qa_chain(vectorstore)

log.info("API initialized successfully. Ready to serve questions.")

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

    # ── Step 1: Semantic search ───────────────────────────────
    steps.append(f"Searching vector database (top_k={request.top_k or TOP_K})...")
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=request.top_k or TOP_K
    )

    if not docs_with_scores:
        steps.append("❌ No relevant chunks found.")
        return QuestionResponse(
            question=question,
            answer="This information is not available in the internal documents.",
            found_in_docs=False,
            sources=[],
            steps=steps,
            duration_seconds=round(time() - start_time, 2),
        )

    # ── Step 2: Log retrieved chunks ─────────────────────────
    for doc, score in docs_with_scores:
        src = doc.metadata.get("source", "unknown")
        steps.append(f"  📄 {src} — relevance score: {score:.3f}")

    # ── Step 3: Generate answer ──────────────────────────────
    steps.append("Generating answer via LLM...")
    result = qa_chain.invoke({"query": question})
    answer = result["result"].strip()
    source_docs = result.get("source_documents", [])

    # ── Step 4: Prepare sources ──────────────────────────────
    seen = set()
    sources = []
    score_map = {id(doc): score for doc, score in docs_with_scores}

    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n"," ").strip() + "..."
        key = (src, preview[:40])
        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(
                document=src,
                chunk_preview=preview,
                relevance_score=round(score_map.get(id(doc),0.0), 3)
            ))

    # ── Step 5: Check if answer contains valid info ─────────
    not_found_phrases = [
        "not available",
        "cannot find",
        "no information",
        "not found",
    ]
    found_in_docs = not any(p.lower() in answer.lower() for p in not_found_phrases)
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