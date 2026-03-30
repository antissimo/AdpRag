# tests/test_api.py

"""
Tests for api.py logic — found_in_docs determination and dynamic top_k.
These are extracted as pure logic tests so no FastAPI server is needed.
"""

import pytest


# ── found_in_docs logic ───────────────────────────────────────────────────

NOT_FOUND_PHRASES = [
    "not available in the internal documents",
    "cannot find",
    "no information",
    "not found in the documents",
    "not mentioned in",
    "no relevant information",
]


def check_found_in_docs(answer: str, sources_count: int) -> bool:
    """Mirrors the found_in_docs logic in api.py."""
    llm_says_not_found = any(p in answer.lower() for p in NOT_FOUND_PHRASES)
    return not llm_says_not_found and sources_count > 0


class TestFoundInDocs:

    def test_found_when_answer_and_sources_present(self):
        assert check_found_in_docs("Expense claims must be submitted within 30 days.", 2) is True

    def test_not_found_when_llm_says_not_available(self):
        assert check_found_in_docs("This information is not available in the internal documents.", 3) is False

    def test_not_found_when_no_sources_even_if_answer_exists(self):
        """Sources=0 means nothing was cited — treat as not found."""
        assert check_found_in_docs("The deadline is 30 days.", 0) is False

    def test_not_found_when_both_conditions_fail(self):
        assert check_found_in_docs("This information is not available in the internal documents.", 0) is False

    def test_not_found_on_cannot_find_phrase(self):
        assert check_found_in_docs("I cannot find this information in the provided context.", 1) is False

    def test_not_found_on_no_relevant_information_phrase(self):
        assert check_found_in_docs("There is no relevant information about this topic.", 1) is False

    def test_found_with_single_source(self):
        assert check_found_in_docs("Travel must be approved by the department manager.", 1) is True

    def test_case_insensitive_matching(self):
        assert check_found_in_docs("This Information Is NOT AVAILABLE IN THE INTERNAL DOCUMENTS.", 2) is False


# ── dynamic top_k logic ───────────────────────────────────────────────────

SIMPLE_TOP_K  = 3
COMPLEX_TOP_K = 8


def get_top_k(complexity: str, override: int | None = None) -> int:
    """Mirrors the get_top_k logic in api.py."""
    if override is not None:
        return override
    return COMPLEX_TOP_K if complexity == "complex" else SIMPLE_TOP_K


class TestDynamicTopK:

    def test_simple_question_gets_small_top_k(self):
        assert get_top_k("simple") == SIMPLE_TOP_K

    def test_complex_question_gets_large_top_k(self):
        assert get_top_k("complex") == COMPLEX_TOP_K

    def test_override_respected_regardless_of_complexity(self):
        assert get_top_k("simple",  override=10) == 10
        assert get_top_k("complex", override=2)  == 2

    def test_unknown_complexity_defaults_to_simple(self):
        assert get_top_k("unknown") == SIMPLE_TOP_K

    def test_none_override_does_not_affect_result(self):
        assert get_top_k("complex", override=None) == COMPLEX_TOP_K