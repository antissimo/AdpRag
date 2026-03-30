# tests/conftest.py

"""
Shared fixtures and helpers available to all test files.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import pytest
from langchain_core.documents import Document


def make_doc(content: str, source: str = "policy.md", quality_score: float = 0.9) -> Document:
    """Create a LangChain Document with metadata for testing."""
    return Document(
        page_content=content,
        metadata={"source": source, "quality_score": quality_score},
    )


# Make make_doc available as a pytest fixture too
@pytest.fixture
def doc_factory():
    return make_doc