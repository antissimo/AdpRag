from langchain_community.llms import Ollama
from .config import OLLAMA_MODEL, TEMPERATURE
from .logger import FileLogger as log

class RAGLLM:
    _instance = None

    @classmethod
    def get(cls) -> Ollama:
        if cls._instance is None:
            log.info(f"Loading LLM ({OLLAMA_MODEL})...")
            cls._instance = Ollama(model=OLLAMA_MODEL, temperature=TEMPERATURE)
            log.info("LLM ready")
        return cls._instance