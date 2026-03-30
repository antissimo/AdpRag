# src/AdpRag/api.py

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from time import time

from AdpRag.vector_store import RAGVectorStore
from AdpRag.qa import create_qa_chain
from AdpRag.reranker import RAGReranker
from AdpRag.logger import FileLogger as log
from AdpRag.config import CHROMA_DIR
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
    full_text: str
    relevance_score: float | None = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    found_in_docs: bool
    sources: list[SourceInfo]
    steps: list[str]
    duration_seconds: float


# ── Dynamic top_k based on complexity ────────────────────────────────────
SIMPLE_TOP_K  = 3
COMPLEX_TOP_K = 8


def get_top_k(complexity: str, override: int | None = None) -> int:
    if override is not None:
        return override
    return COMPLEX_TOP_K if complexity == "complex" else SIMPLE_TOP_K


# ── Init ──────────────────────────────────────────────────────────────────
if not Path(CHROMA_DIR).exists():
    raise RuntimeError(f"ChromaDB does not exist ({CHROMA_DIR}). Run setup first!")

rag_store   = RAGVectorStore(chroma_dir=CHROMA_DIR)
vectorstore = rag_store.load_vectorstore()

qa_chain = create_qa_chain()
reranker = RAGReranker.get()
agent    = RAGAgent(vectorstore)

log.info("API v2 initialized (agentic mode). Ready to serve questions.")


# ── Helper: build sources list ────────────────────────────────────────────
def _build_sources(source_docs, docs_with_scores) -> list[SourceInfo]:
    seen      = set()
    sources   = []
    score_map = {hash(doc.page_content[:100]): score for doc, score in docs_with_scores}

    for doc in source_docs:
        src     = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ").strip() + "..."
        key     = (src, preview[:40])

        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(
                document=src,
                chunk_preview=preview,
                full_text=doc.page_content,
                relevance_score=round(score_map.get(hash(doc.page_content[:100]), 0.0), 3),
            ))

    return sources


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "ok",
        "message": "Internal Docs Q&A API (Agentic RAG)",
        "version": "2.0.0",
        "endpoints": {
            "ask":        "POST /ask",
            "ask/stream": "POST /ask/stream",
            "health":     "GET /health",
            "docs":       "GET /docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "chunks_in_db": vectorstore._collection.count(),
        "chroma_dir":   str(CHROMA_DIR),
    }


# ── POST /ask  (standard — full response at once) ─────────────────────────
@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    start_time = time()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    agent_result     = agent.run(question)
    steps            = agent_result["steps"]
    docs_with_scores = agent_result["docs_with_scores"]
    complexity       = agent_result["complexity"]

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

    top_k = get_top_k(complexity, override=request.top_k)
    steps.append(f"⚖️  [Reranker] Complexity={complexity} → top_k={top_k}. Reranking {len(docs_with_scores)} chunks...")

    try:
        docs_with_scores = reranker.rerank(question, docs_with_scores, top_k=top_k)
        for doc, score in docs_with_scores:
            steps.append(f"   📄 {doc.metadata.get('source', 'unknown')} — final score: {score:.3f}")
    except Exception as e:
        log.warning(f"Reranker failed: {e}")
        steps.append(f"   ⚠️  Reranker failed ({e}), using unranked docs")

    steps.append("💬 [LLM] Generating answer from collected context...")
    reranked_docs = [doc for doc, _ in docs_with_scores]
    result        = qa_chain.invoke(question, reranked_docs)
    answer        = result["result"]
    source_docs   = result["source_documents"]

    sources       = _build_sources(source_docs, docs_with_scores)
    found_in_docs = len(sources) > 0

    steps.append(f"✅ [Done] found_in_docs={found_in_docs} (sources={len(sources)})")
    duration = round(time() - start_time, 2)
    steps.append(f"⏱️  Total duration: {duration}s")

    return QuestionResponse(
        question=question,
        answer=answer,
        found_in_docs=found_in_docs,
        sources=sources,
        steps=steps,
        duration_seconds=duration,
    )


# ── POST /ask/stream  (SSE — steps arrive one by one) ─────────────────────
@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_generator():
        start_time = time()

        def sse(event_type: str, data: dict) -> str:
            return f"data: {json.dumps({'type': event_type, **data})}\n\n"

        agent_result     = agent.run(question)
        docs_with_scores = agent_result["docs_with_scores"]
        complexity       = agent_result["complexity"]

        for step in agent_result["steps"]:
            yield sse("step", {"text": step})

        if not docs_with_scores:
            yield sse("step", {"text": "❌ No relevant chunks found after all iterations."})
            yield sse("done", {
                "answer":           "This information is not available in the internal documents.",
                "found_in_docs":    False,
                "sources":          [],
                "duration_seconds": round(time() - start_time, 2),
            })
            return

        top_k = get_top_k(complexity, override=request.top_k)
        yield sse("step", {"text": f"⚖️  [Reranker] Complexity={complexity} → top_k={top_k}. Reranking {len(docs_with_scores)} chunks..."})

        try:
            docs_with_scores = reranker.rerank(question, docs_with_scores, top_k=top_k)
            for doc, score in docs_with_scores:
                yield sse("step", {"text": f"   📄 {doc.metadata.get('source', 'unknown')} — final score: {score:.3f}"})
        except Exception as e:
            log.warning(f"Reranker failed: {e}")
            yield sse("step", {"text": f"   ⚠️  Reranker failed ({e}), using unranked docs"})

        yield sse("step", {"text": "💬 [LLM] Generating answer from collected context..."})
        reranked_docs = [doc for doc, _ in docs_with_scores]
        result        = qa_chain.invoke(question, reranked_docs)
        answer        = result["result"]
        source_docs   = result["source_documents"]

        sources       = _build_sources(source_docs, docs_with_scores)
        found_in_docs = len(sources) > 0

        duration = round(time() - start_time, 2)
        yield sse("step", {"text": f"✅ [Done] found_in_docs={found_in_docs} (sources={len(sources)})"})
        yield sse("step", {"text": f"⏱️  Total duration: {duration}s"})

        yield sse("done", {
            "answer":           answer,
            "found_in_docs":    found_in_docs,
            "sources":          [s.model_dump() for s in sources],
            "duration_seconds": duration,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )