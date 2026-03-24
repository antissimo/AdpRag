from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import EMBED_MODEL, CHROMA_DIR
from .logger import FileLogger as log  


class RAGVectorStore:
    def __init__(self, chroma_dir=CHROMA_DIR, embed_model=EMBED_MODEL):
        self.chroma_dir = chroma_dir
        self.embed_model = embed_model

    def create_vectorstore(self, chunks):
        log.info(f"Loading embedding model ({self.embed_model})...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={"device": "cpu"},
        )
        log.info("Embedding model ready")

        log.info(f"Saving to ChromaDB ({self.chroma_dir})...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.chroma_dir),
        )
        log.info(f"Vector database created with {vectorstore._collection.count()} chunks")
        return vectorstore
    def load_vectorstore(self):
        log.info(f"Loading existing ChromaDB from {self.chroma_dir}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={"device": "cpu"},
        )
        vectorstore = Chroma(
            persist_directory=str(self.chroma_dir),
            embedding_function=embeddings,
        )
        log.info(f"Vector database loaded with {vectorstore._collection.count()} chunks")
        return vectorstore