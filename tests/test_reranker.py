# tests/test_reranker.py

"""
Tests for reranker.py — scoring formula, top-K selection, MIN_RELEVANCE filter.
CrossEncoder is mocked so no model loading is needed.
"""

import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import make_doc


@pytest.fixture(autouse=True)
def fresh_reranker():
    """Reset the RAGReranker singleton before each test."""
    from AdpRag import reranker as reranker_module
    reranker_module.RAGReranker._instance = None
    yield
    reranker_module.RAGReranker._instance = None


@pytest.fixture
def reranker(fresh_reranker):
    with patch("AdpRag.reranker.CrossEncoder") as mock_ce_cls:
        mock_ce = MagicMock()
        mock_ce_cls.return_value = mock_ce
        from AdpRag.reranker import RAGReranker
        instance = RAGReranker()
        instance._mock_ce = mock_ce  # expose for tests
        return instance


class TestRerankerScoring:

    def test_returns_empty_on_no_docs(self, reranker):
        assert reranker.rerank("any question", []) == []

    def test_high_quality_doc_beats_low_quality_with_same_ce_score(self, reranker):
        """
        Two docs with identical CrossEncoder score.
        Higher quality_score should rank first.
        """
        doc_high = make_doc("expense policy content", "expense_policy.md", quality_score=0.95)
        doc_low  = make_doc("expense wip content",    "expense_wip.md",    quality_score=0.30)

        reranker._mock_ce.predict.return_value = [1.0, 1.0]

        result     = reranker.rerank("expense deadline", [(doc_high, 0.8), (doc_low, 0.8)], top_k=2)
        top_doc, _ = result[0]
        assert top_doc == doc_high

    def test_relevant_wip_doc_beats_irrelevant_official_doc(self, reranker):
        """
        A WIP doc directly relevant to the query should outrank
        an official doc that has nothing to do with the query.
        """
        doc_relevant_wip = make_doc("expense claim 30 day deadline", "expense_wip.md",  quality_score=0.3)
        doc_irrelevant   = make_doc("general company overview intro", "hr_handbook.md", quality_score=0.95)

        reranker._mock_ce.predict.return_value = [5.0, -2.0]

        result     = reranker.rerank("expense deadline", [(doc_relevant_wip, 0.5), (doc_irrelevant, 0.5)], top_k=2)
        top_doc, _ = result[0]
        assert top_doc == doc_relevant_wip

    def test_results_sorted_descending_by_score(self, reranker):
        docs = [(make_doc(f"doc {i}", quality_score=0.5), 0.5) for i in range(5)]
        reranker._mock_ce.predict.return_value = [1.0, 3.0, 2.0, 5.0, 4.0]

        result = reranker.rerank("query", docs, top_k=5)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, reranker):
        docs = [(make_doc(f"doc {i}", quality_score=0.8), 0.5) for i in range(10)]
        reranker._mock_ce.predict.return_value = [float(i) for i in range(10)]

        result = reranker.rerank("query", docs, top_k=3)
        assert len(result) <= 3

    def test_filters_chunks_below_min_relevance(self, reranker):
        """Chunks with very low CE score should be dropped entirely."""
        doc = make_doc("barely relevant content", quality_score=0.1)
        reranker._mock_ce.predict.return_value = [-10.0]

        result = reranker.rerank("query", [(doc, 0.5)], top_k=5)
        assert len(result) == 0

    def test_single_doc_returned_correctly(self, reranker):
        doc = make_doc("travel approval requires manager sign-off", quality_score=0.9)
        reranker._mock_ce.predict.return_value = [2.0]

        result = reranker.rerank("travel approval", [(doc, 0.7)], top_k=5)
        assert len(result) == 1
        assert result[0][0] == doc