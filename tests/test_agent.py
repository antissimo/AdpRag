# tests/test_agent.py

"""
Tests for agent.py — _merge_docs deduplication, loop guard, and _plan fallback.
LLM and vector store are fully mocked.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import make_doc


@pytest.fixture
def agent():
    """Return a RAGAgent instance with mocked LLM and vector store."""
    with patch("AdpRag.agent.RAGLLM.get", return_value=MagicMock()):
        from AdpRag.agent import RAGAgent
        instance = RAGAgent.__new__(RAGAgent)
        instance.llm = MagicMock()
        instance.vectorstore = MagicMock()
        instance.vectorstore.similarity_search_with_relevance_scores.return_value = []
        return instance


class TestAgentMergeDocs:
    """Tests for chunk deduplication logic."""

    def test_deduplicates_identical_chunks(self, agent):
        doc  = make_doc("same content")
        seen = {doc.page_content}
        result = agent._merge_docs([(doc, 0.5)], [(doc, 0.5)], seen, [])
        assert len(result) == 1

    def test_keeps_higher_score_on_duplicate(self, agent):
        doc  = make_doc("same content")
        seen = set()

        result = agent._merge_docs([], [(doc, 0.3)], seen, [])
        result = agent._merge_docs(result, [(doc, 0.9)], seen, [])

        assert len(result) == 1
        assert result[0][1] == 0.9

    def test_adds_new_unique_chunks(self, agent):
        doc1 = make_doc("first document content")
        doc2 = make_doc("second document content")
        seen = set()

        result = agent._merge_docs([], [(doc1, 0.5)], seen, [])
        result = agent._merge_docs(result, [(doc2, 0.7)], seen, [])

        assert len(result) == 2

    def test_preserves_existing_when_no_new_docs(self, agent):
        doc    = make_doc("existing content")
        seen   = {doc.page_content}
        result = agent._merge_docs([(doc, 0.8)], [], seen, [])
        assert len(result) == 1
        assert result[0][1] == 0.8

    def test_step_logged(self, agent):
        steps = []
        agent._merge_docs([], [(make_doc("content"), 0.5)], set(), steps)
        assert any("new unique chunks" in s for s in steps)


class TestAgentLoopGuard:
    """Tests that the agent stops when LLM suggests a repeated query."""

    def test_stops_on_repeated_query(self, agent):
        repeated_query = "expense policy deadline"

        agent._plan = MagicMock(return_value={
            "complexity": "simple",
            "reasoning":  "test",
            "queries":    [repeated_query],
        })
        agent._evaluate = MagicMock(return_value={
            "enough":     False,
            "missing":    "deadline info",
            "next_query": repeated_query,  # same as already tried!
        })

        result = agent.run("What is the expense deadline?")
        assert any("Stopping to avoid loop" in s for s in result["steps"])

    def test_stops_when_no_next_query_suggested(self, agent):
        agent._plan = MagicMock(return_value={
            "complexity": "simple",
            "reasoning":  "test",
            "queries":    ["expense deadline"],
        })
        agent._evaluate = MagicMock(return_value={
            "enough":     False,
            "missing":    "some info",
            "next_query": None,
        })

        result = agent.run("What is the expense deadline?")
        assert any("No follow-up query suggested" in s for s in result["steps"])

    def test_stops_after_max_iterations(self, agent):
        """Agent must never exceed MAX_AGENT_ITERATIONS."""
        from AdpRag.config import MAX_AGENT_ITERATIONS

        agent._plan = MagicMock(return_value={
            "complexity": "simple",
            "reasoning":  "test",
            "queries":    ["query one"],
        })

        call_count = 0

        def evaluate_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"enough": False, "missing": "info", "next_query": f"unique query {call_count}"}

        agent._evaluate = MagicMock(side_effect=evaluate_side_effect)

        agent.run("Any question?")
        assert call_count <= MAX_AGENT_ITERATIONS


class TestAgentPlanFallback:
    """Tests that planning degrades gracefully when LLM fails or returns garbage."""

    def test_falls_back_on_llm_exception(self, agent):
        agent.llm.invoke.side_effect = Exception("Ollama timeout")
        question = "What is the travel approval policy?"
        result   = agent._plan(question)
        assert result["queries"]    == [question]
        assert result["complexity"] == "simple"

    def test_falls_back_when_no_json_in_response(self, agent):
        agent.llm.invoke.return_value = "I cannot determine the search strategy."
        question = "Who approves travel?"
        result   = agent._plan(question)
        assert result["queries"] == [question]

    def test_falls_back_when_queries_list_empty(self, agent):
        agent.llm.invoke.return_value = json.dumps({
            "complexity": "simple",
            "reasoning":  "test",
            "queries":    [],
        })
        question = "test question"
        result   = agent._plan(question)
        assert result["queries"] == [question]

    def test_complexity_returned_in_run_result(self, agent):
        agent._plan = MagicMock(return_value={
            "complexity": "complex",
            "reasoning":  "needs multiple docs",
            "queries":    ["query one"],
        })
        agent._evaluate = MagicMock(return_value={
            "enough": True, "missing": "nothing", "next_query": None
        })

        result = agent.run("Create onboarding checklist")
        assert result["complexity"] == "complex"