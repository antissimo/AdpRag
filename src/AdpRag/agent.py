# src/AdpRag/agent.py

from .llm import RAGLLM
from .logger import FileLogger as log
from .instructions import AGENT_PLAN_PROMPT, AGENT_EVALUATE_PROMPT
from .config import MAX_AGENT_ITERATIONS, MAX_QUERIES_PER_ITERATION, TOP_K


class RAGAgent:
    """
    Agentic retrieval loop:
      1. Plans how many queries to fire based on question complexity
      2. Executes queries against the vector store
      3. Evaluates if collected docs are sufficient
      4. If not, generates a NEW (different) query and repeats
      5. Returns all collected docs + complexity + detailed steps for frontend display
    """

    _instance = None

    @classmethod
    def get(cls, vectorstore):
        if cls._instance is None:
            cls._instance = cls(vectorstore)
        return cls._instance

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = RAGLLM.get()
        log.info("RAGAgent initialized")

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, question: str) -> dict:
        """
        Run the full agentic retrieval for a question.

        Returns:
            {
                "docs_with_scores": [...],   # (Document, float) pairs
                "complexity": "simple" | "complex",
                "steps": [...],              # human-readable steps for frontend
            }
        """
        steps: list[str]         = []
        all_docs_with_scores     = []
        seen_contents: set[str]  = set()
        tried_queries: list[str] = []

        # ── Phase 1: Planning ─────────────────────────────────────────────
        steps.append("🤖 [Agent] Analyzing question complexity and planning search strategy...")
        plan = self._plan(question)

        complexity = plan.get("complexity", "simple")
        reasoning  = plan.get("reasoning", "")
        queries    = plan.get("queries", [question])[:MAX_QUERIES_PER_ITERATION]

        steps.append(f"🧠 [Agent] Complexity: {complexity.upper()} — {reasoning}")
        steps.append(f"📋 [Agent] Initial queries planned: {len(queries)}")
        for i, q in enumerate(queries, 1):
            steps.append(f"   Query {i}: \"{q}\"")

        # ── Phase 2: Initial retrieval ────────────────────────────────────
        steps.append("🔍 [Agent] Executing initial queries against vector store...")
        new_docs = self._execute_queries(queries, steps)
        tried_queries.extend(queries)
        all_docs_with_scores = self._merge_docs(all_docs_with_scores, new_docs, seen_contents, steps)

        # ── Phase 3: Iterative retrieval loop ─────────────────────────────
        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            steps.append(f"🔄 [Agent] Iteration {iteration}/{MAX_AGENT_ITERATIONS} — evaluating collected context...")

            evaluation = self._evaluate(
                question=question,
                docs_with_scores=all_docs_with_scores,
                iteration=iteration,
                max_iterations=MAX_AGENT_ITERATIONS,
                previous_queries=tried_queries,
            )

            enough     = evaluation.get("enough", True)
            missing    = evaluation.get("missing", "nothing")
            next_query = evaluation.get("next_query")

            if enough:
                steps.append(f"✅ [Agent] Context is sufficient after iteration {iteration}. Proceeding to answer generation.")
                break

            steps.append(f"⚠️  [Agent] Missing info: \"{missing}\"")

            if not next_query:
                steps.append("⚠️  [Agent] No follow-up query suggested. Stopping iterations.")
                break

            if next_query in tried_queries:
                steps.append(f"⚠️  [Agent] Suggested query already tried: \"{next_query}\". Stopping to avoid loop.")
                break

            steps.append(f"➕ [Agent] Follow-up query: \"{next_query}\"")
            extra_docs = self._execute_queries([next_query], steps)
            tried_queries.append(next_query)
            all_docs_with_scores = self._merge_docs(all_docs_with_scores, extra_docs, seen_contents, steps)

        steps.append(f"📦 [Agent] Total unique chunks collected: {len(all_docs_with_scores)}")

        return {
            "docs_with_scores": all_docs_with_scores,
            "complexity":       complexity,   # expose to api.py for dynamic top_k
            "steps":            steps,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _plan(self, question: str) -> dict:
        """Ask LLM to plan the search strategy."""
        import json, re
        prompt = AGENT_PLAN_PROMPT.format(question=question)
        try:
            response = self.llm.invoke(prompt)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data    = json.loads(match.group(0))
                queries = data.get("queries", [question])
                if not isinstance(queries, list) or not queries:
                    queries = [question]
                return {
                    "complexity": data.get("complexity", "simple"),
                    "reasoning":  data.get("reasoning", ""),
                    "queries":    queries,
                }
        except Exception as e:
            log.warning(f"Agent planning failed: {e}. Falling back to single query.")

        return {"complexity": "simple", "reasoning": "fallback", "queries": [question]}

    def _evaluate(
        self,
        question: str,
        docs_with_scores: list,
        iteration: int,
        max_iterations: int,
        previous_queries: list[str],
    ) -> dict:
        """Ask LLM if we have enough context or need another query."""
        import json, re

        preview_parts = []
        for doc, score in docs_with_scores[:6]:
            src     = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:200].replace("\n", " ")
            preview_parts.append(f"[{src}] {snippet}...")
        context_preview = "\n".join(preview_parts) if preview_parts else "No documents collected yet."

        prompt = AGENT_EVALUATE_PROMPT.format(
            question=question,
            context_preview=context_preview,
            iteration=iteration,
            max_iterations=max_iterations,
            previous_queries=", ".join(f'"{q}"' for q in previous_queries),
        )

        try:
            response = self.llm.invoke(prompt)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "enough":     bool(data.get("enough", True)),
                    "missing":    data.get("missing", "nothing"),
                    "next_query": data.get("next_query") or None,
                }
        except Exception as e:
            log.warning(f"Agent evaluation failed: {e}. Assuming sufficient context.")

        return {"enough": True, "missing": "nothing", "next_query": None}

    def _execute_queries(self, queries: list[str], steps: list) -> list[tuple]:
        """Run one or more queries against the vector store."""
        results = []
        for query in queries:
            try:
                hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=TOP_K)
                steps.append(f"   🗄️  Vector search: \"{query}\" → {len(hits)} chunks retrieved")
                results.extend(hits)
            except Exception as e:
                log.warning(f"Vector search failed for query '{query}': {e}")
                steps.append(f"   ❌ Vector search failed: \"{query}\" — {e}")
        return results

    def _merge_docs(
        self,
        existing: list[tuple],
        new_docs: list[tuple],
        seen: set,
        steps: list,
    ) -> list[tuple]:
        """Merge new docs into existing, deduplicating by content, keeping highest score."""
        score_map: dict[str, tuple] = {doc.page_content: (doc, score) for doc, score in existing}
        added = 0

        for doc, score in new_docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                score_map[key] = (doc, score)
                added += 1
            else:
                if score > score_map[key][1]:
                    score_map[key] = (doc, score)

        steps.append(f"   ✨ {added} new unique chunks added (duplicates merged)")
        return list(score_map.values())