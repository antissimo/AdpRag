# embedder.py
from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBED_MODEL
from .logger import FileLogger as log

class RAGEmbedder:
    _instance = None

    @classmethod
    def get(cls, embed_model=EMBED_MODEL) -> HuggingFaceEmbeddings:
        if cls._instance is None:
            log.info(f"Loading embedding model ({embed_model})...")
            cls._instance = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs={"device": "cpu"},
            )
            log.info("Embedding model ready")
        return cls._instance