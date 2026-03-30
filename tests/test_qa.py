# tests/test_qa.py

"""
Tests for qa.py — SimpleQAChain answer generation.
LLM is mocked so no Ollama instance is needed.
"""

import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import make_doc


@pytest.fixture
def qa_chain():
    with patch("AdpRag.qa.RAGLLM.get") as mock_llm_get:
        mock_llm = MagicMock()
        mock_llm_get.return_value = mock_llm
        from AdpRag.qa import SimpleQAChain
        chain = SimpleQAChain()
        chain._mock_llm = mock_llm  # expose for assertions
        return chain


class TestSimpleQAChain:

    def test_returns_result_and_source_documents(self, qa_chain):
        qa_chain._mock_llm.invoke.return_value = "The deadline is 30 days."
        docs   = [make_doc("Expense claims must be filed within 30 days.")]
        result = qa_chain.invoke("What is the deadline?", docs)

        assert "result"           in result
        assert "source_documents" in result
        assert result["result"]           == "The deadline is 30 days."
        assert result["source_documents"] == docs

    def test_combines_multiple_docs_into_context(self, qa_chain):
        qa_chain._mock_llm.invoke.return_value = "Answer."
        docs = [
            make_doc("First chunk content."),
            make_doc("Second chunk content."),
        ]
        qa_chain.invoke("question", docs)

        called_prompt = qa_chain._mock_llm.invoke.call_args[0][0]
        assert "First chunk content."  in called_prompt
        assert "Second chunk content." in called_prompt

    def test_question_included_in_prompt(self, qa_chain):
        qa_chain._mock_llm.invoke.return_value = "Answer."
        qa_chain.invoke("What is the travel policy?", [make_doc("some content")])

        called_prompt = qa_chain._mock_llm.invoke.call_args[0][0]
        assert "What is the travel policy?" in called_prompt

    def test_empty_docs_still_invokes_llm(self, qa_chain):
        qa_chain._mock_llm.invoke.return_value = "This information is not available in the internal documents."
        result = qa_chain.invoke("Any question?", [])

        assert result["source_documents"] == []
        qa_chain._mock_llm.invoke.assert_called_once()

    def test_source_documents_match_input_docs(self, qa_chain):
        qa_chain._mock_llm.invoke.return_value = "Answer."
        docs   = [make_doc(f"chunk {i}") for i in range(3)]
        result = qa_chain.invoke("question", docs)

        assert result["source_documents"] == docs