# AdpRag


INGESTION (ingestion.py)

Loading .md files → Semantic chunking → Embedding (all-MiniLM-L6-v2) → Storing into vector DB

Important setting: CHUNKING_THRESHOLD

PROMPTING

User prompt → query transform → vector search → rerank → LLM