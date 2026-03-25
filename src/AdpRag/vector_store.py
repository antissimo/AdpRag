from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import CHROMA_DIR
from .logger import FileLogger as log  
from .embedder import RAGEmbedder


class RAGVectorStore:
    def __init__(self, chroma_dir=CHROMA_DIR):
        self.chroma_dir = chroma_dir

    def create_vectorstore(self, chunks):
        log.info(f"Saving to ChromaDB ({self.chroma_dir})...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=RAGEmbedder.get(),
            persist_directory=str(self.chroma_dir),
        )
        log.info(f"Vector database created with {vectorstore._collection.count()} chunks")
        return vectorstore
    def load_vectorstore(self):
        log.info(f"Loading existing ChromaDB from {self.chroma_dir}...")
        vectorstore = Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=RAGEmbedder.get(),
                collection_metadata={"hnsw:space": "cosine"}, 
            )
        log.info(f"Vector database loaded with {vectorstore._collection.count()} chunks")
        return vectorstore