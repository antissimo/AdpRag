from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Folders ───────────────────────────────────────────────────────────────

DOCUMENTS_DIR = PROJECT_ROOT / "documents"   # place your .md files here
CHROMA_DIR    = PROJECT_ROOT / "chroma_db"   # vector database (auto-created on ingestion)
LOG_FILE      = PROJECT_ROOT / "logs/log.txt"
CACHE_DIR     = PROJECT_ROOT / ".cache"


# ── Models ────────────────────────────────────────────────────────────────

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # embedding model (HuggingFace)
OLLAMA_MODEL = "mistral"                                  # LLM served via Ollama
TEMPERATURE  = 0.1   # low = more deterministic answers, high = more creative


# ── Ingestion ─────────────────────────────────────────────────────────────

# Controls how aggressively SemanticChunker splits documents.
# Lower value = more chunks (finer granularity).
# Higher value = fewer but larger chunks.
# Range: 0–100 (percentile). Recommended: 60–85.
CHUNKING_THRESHOLD = 70

# Chunks with a combined quality score below this threshold are dropped
# during ingestion and never stored in the vector database.
# Quality score = (content_score × 0.7) + (doc_score × 0.3)
# Range: 0.0–1.0. Raise to be stricter, lower to be more permissive.
QUALITY_DROP_THRESHOLD = 0.25


# ── Retrieval ─────────────────────────────────────────────────────────────

# Number of chunks fetched from the vector store per search query.
# Higher = more candidates for reranking, but slower.
TOP_K = 15

# Minimum final reranker score for a chunk to be kept.
# Chunks scoring below this are discarded before answer generation.
# Range: 0.0–1.0.
MIN_RELEVANCE = 0.3


# ── Agent ─────────────────────────────────────────────────────────────────

# Maximum number of iterative retrieval rounds the agent can perform.
# Each round: evaluate context → decide if more info needed → new query.
# Higher = more thorough but slower. Recommended: 2–4.
MAX_AGENT_ITERATIONS = 2

# Maximum number of search queries the agent fires in the initial planning phase.
# Simple questions use 1, complex questions use up to this many.
# Recommended: 2–4.
MAX_QUERIES_PER_ITERATION = 2


# ── Reranking ─────────────────────────────────────────────────────────────

# Weights for the final reranking score formula:
#   final_score = (ce_score × CROSS_ENCODER_WEIGHT) + (quality_score × QUALITY_SCORE_WEIGHT)
# CrossEncoder score reflects query-chunk relevance.
# Quality score reflects document trustworthiness (pre-computed at ingestion).

# Cross encoder returns value [-10,10]
# Quality score is [0,1], so we give it more weight to have a stronger impact on final ranking.

CROSS_ENCODER_WEIGHT = 0.7
QUALITY_SCORE_WEIGHT = 3