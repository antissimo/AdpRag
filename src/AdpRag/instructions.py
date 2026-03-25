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

