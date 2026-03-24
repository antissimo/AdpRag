import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import shutil
from pathlib import Path

from AdpRag.logger import FileLogger as log
from AdpRag.loader import RAGLoader
from AdpRag.vector_store import RAGVectorStore
from AdpRag.config import CHROMA_DIR

def main():
    log.info("Starting setup...")

    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        log.info("Deleting existing ChromaDB...")
        shutil.rmtree(chroma_path)

    loader = RAGLoader()
    chunks = loader.load_documents()
    if not chunks:
        log.error("There are no documents to process.")
        return
    
    vectorstore = RAGVectorStore()
    vectorstore.create_vectorstore(chunks)

    log.info("Setup completed successfully!")
    log.info(f"ChromaDB: {CHROMA_DIR}")
    log.info(f"Number of chunks: {len(chunks)}")

if __name__ == "__main__":
    main()