# tests/test_helpers.py

"""
Tests for helpers.py — parse_llm_json utility.
No mocks needed, pure function tests.
"""

import pytest
from AdpRag.helpers import parse_llm_json


class TestParseLlmJson:
    """Tests for the JSON parsing utility used across the system."""

    def test_parses_clean_json(self):
        result = parse_llm_json('{"priority": 0.8, "reason": "official document"}')
        assert result["priority"] == 0.8
        assert result["reason"] == "official document"

    def test_extracts_json_from_surrounding_text(self):
        """LLMs often add preamble before JSON — should still parse."""
        result = parse_llm_json('Here is my assessment: {"priority": 0.6, "reason": "draft"} Thank you.')
        assert result["priority"] == 0.6

    def test_converts_string_priority_to_float(self):
        result = parse_llm_json('{"priority": "0.7", "reason": "ok"}')
        assert isinstance(result["priority"], float)
        assert result["priority"] == 0.7

    def test_falls_back_priority_on_invalid_value(self):
        result = parse_llm_json('{"priority": "not_a_number", "reason": "test"}')
        assert result["priority"] == 0.5
    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            parse_llm_json("")

    def test_handles_multiline_json(self):
        text = """
        {
            "priority": 0.9,
            "reason": "finalized policy document"
        }
        """
        result = parse_llm_json(text)
        assert result["priority"] == 0.9