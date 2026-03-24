from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .logger import FileLogger as log

class RAGLoader:
    def __init__(self, documents_dir=DOCUMENTS_DIR):
        self.documents_dir = Path(documents_dir)

    def load_documents(self):
        if not self.documents_dir.exists():
            log.error(f"Folder '{self.documents_dir}' does not exist!")
            return []

        md_files = list(self.documents_dir.glob("**/*.md"))
        if not md_files:
            log.warning(f"There are no .md files in '{self.documents_dir}'!")
            return []

        documents = DirectoryLoader(
            str(self.documents_dir),
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True
        ).load()

        for doc in documents:
            doc.metadata["source"] = Path(doc.metadata.get("source", "")).name

        log.info(f"Loaded {len(documents)} pages/sections")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(documents)
        log.info(f"Created {len(chunks)} chunks")
        return chunks