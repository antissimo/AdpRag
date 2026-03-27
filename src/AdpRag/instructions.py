PROMPT_TEMPLATE = """You are an assistant that answers questions ONLY based on the internal company documents provided.

Rules:
- Answer ONLY using the context below.
- If the information is not in the context, reply exactly: "This information is not available in the internal documents."
- Be concise and precise.
- Answer in the same language as the question.

Context from documents:
{context}

Question: {question}

Answer:"""


RERANKING_PROMPT = """
You are ranking document reliability and authority.

You are ONLY given the document filename.
You MUST infer how final, official, and trustworthy it is.

Return ONLY JSON:
{{"priority": <float 0.0-1.0>, "reason": "<one sentence>"}}

Scoring rules:
- 1.0 → official, finalized, authoritative document (approved policy)
- 0.7–0.9 → mostly reliable
- 0.4–0.6 → draft / unclear
- 0.1–0.3 → low reliability

CRITICAL:
- If filename contains "wip", "draft", "test", "tmp" → priority MUST be <= 0.4
- Prefer clean names like "policy", "official", "final"

Document filename: <<SOURCE>>
"""


# ── Agent prompts ─────────────────────────────────────────────────────────

AGENT_PLAN_PROMPT = """You are a search planning agent for a RAG document system.

Your job is to analyze the user question and decide the search strategy.

Rules:
- If the question is SIMPLE (one specific fact) → use 1 query
- If the question is COMPLEX (multi-step, checklist, comparison, summary) → use 2-3 queries
- Queries should be short, keyword-focused, and cover different aspects of the question
- Return ONLY valid JSON, no explanation, no markdown

Return JSON:
{{
  "complexity": "simple" | "complex",
  "reasoning": "<one sentence why>",
  "queries": ["<query1>", "<query2>", ...]
}}

User question: {question}
"""


AGENT_EVALUATE_PROMPT = """You are evaluating whether collected document chunks are sufficient to answer a question.

Question: {question}

Document chunks collected so far:
{context_preview}

Iteration: {iteration} of {max_iterations}

Rules:
- If you have enough information to answer the question → set "enough" to true
- If important information is clearly missing → set "enough" to false and provide a new search query
- If this is the last iteration → you MUST set "enough" to true regardless
- Return ONLY valid JSON, no explanation, no markdown

Return JSON:
{{
  "enough": true | false,
  "missing": "<what specific information is still missing, or 'nothing' if enough>",
  "next_query": "<focused search query to find missing info, or null if enough>"
}}
"""


def format_reranking_prompt(source: str) -> str:
    try:
        if not source:
            raise ValueError("Source is empty")

        prompt = RERANKING_PROMPT.replace("<<SOURCE>>", str(source))
        return prompt

    except Exception:
        return f"""
                Return JSON:
                {{"priority": 0.5, "reason": "fallback"}}

                Filename: {source}
                """