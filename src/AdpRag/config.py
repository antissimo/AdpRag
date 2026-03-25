from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Folders
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LOG_FILE = PROJECT_ROOT / "logs/log.txt"

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TOP_K = 5
MIN_RELEVANCE = 0.3

#Ingestion settings

CHUNKING_THRESHOLD = 70 #percentile for chunking, LOWER means MORE chunks