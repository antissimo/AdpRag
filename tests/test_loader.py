# tests/test_loader.py

"""
Tests for loader.py — quality response parsing and chunk filtering.
RAGEmbedder is mocked so no model loading is needed.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import make_doc


class TestLoaderParseQualityResponse:
    """
    Tests for the quality response parser in RAGLoader.
    This is the most critical ingestion component — if it fails,
    garbage chunks get a free pass with quality_score=0.5.
    """

    def setup_method(self):
        with patch("AdpRag.loader.RAGEmbedder.get", return_value=MagicMock()):
            from AdpRag.loader import RAGLoader
            self.loader = RAGLoader.__new__(RAGLoader)
            self.parse  = RAGLoader._parse_quality_response.__get__(self.loader)

    def test_parses_full_quality_response(self):
        response = json.dumps({
            "content_score": 0.9,
            "doc_score":     1.0,
            "quality_score": 0.93,
            "reason":        "Clear policy document",
        })
        result = self.parse(response)
        assert result["quality_score"] == 0.93
        assert result["content_score"] == 0.9
        assert result["doc_score"]     == 1.0
        assert result["reason"]        == "Clear policy document"

    def test_returns_fallback_on_invalid_json(self):
        result = self.parse("Sorry, I cannot evaluate this.")
        assert result["quality_score"] == 0.5
        assert result["content_score"] == 0.5
        assert result["doc_score"]     == 0.5
        assert "fallback" in result["reason"] or "parse failed" in result["reason"]

    def test_returns_fallback_on_empty_response(self):
        result = self.parse("")
        assert result["quality_score"] == 0.5

    def test_extracts_json_from_verbose_response(self):
        """LLM might wrap JSON in explanation text."""
        response = 'Sure! Here is my evaluation: {"content_score": 0.1, "doc_score": 0.3, "quality_score": 0.16, "reason": "gibberish"}'
        result = self.parse(response)
        assert result["quality_score"] == 0.16
        assert result["reason"]        == "gibberish"

    def test_garbage_document_gets_low_score(self):
        """Simulate what LLM returns for a garbage document."""
        response = json.dumps({
            "content_score": 0.0,
            "doc_score":     0.5,
            "quality_score": 0.15,
            "reason":        "Pure gibberish, random invented words",
        })
        result = self.parse(response)
        assert result["quality_score"] < 0.25  # must fall below drop threshold


class TestLoaderQualityFilter:
    """Tests for the chunk quality drop threshold logic."""

    THRESHOLD = 0.25

    def _filter(self, chunks):
        return [c for c in chunks if c.metadata.get("quality_score", 1.0) >= self.THRESHOLD]

    def test_drops_chunks_below_threshold(self):
        chunks = [
            make_doc("Good policy content",    quality_score=0.9),
            make_doc("velm zortha quillex",     source="garbage.md", quality_score=0.1),
            make_doc("Another good document",  quality_score=0.8),
            make_doc("Borderline content",     quality_score=0.24),  # just below threshold
        ]
        filtered = self._filter(chunks)
        assert len(filtered) == 2
        assert all(c.metadata["quality_score"] >= self.THRESHOLD for c in filtered)

    def test_keeps_chunks_exactly_at_threshold(self):
        chunks = [make_doc("Exactly at threshold", quality_score=0.25)]
        assert len(self._filter(chunks)) == 1

    def test_all_garbage_results_in_empty_list(self):
        chunks = [
            make_doc("flarn zortha", quality_score=0.05),
            make_doc("velm velm velm", quality_score=0.10),
        ]
        assert len(self._filter(chunks)) == 0

    def test_all_good_docs_pass_through(self):
        chunks = [make_doc(f"Good content {i}", quality_score=0.8) for i in range(5)]
        assert len(self._filter(chunks)) == 5