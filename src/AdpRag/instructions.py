# src/AdpRag/instructions.py

PROMPT_TEMPLATE = """You are an assistant that answers questions ONLY based on the internal company documents provided.

Rules:
- Answer ONLY using the context below. Do NOT use any outside knowledge.
- If the information is not in the context, reply with ONLY this exact sentence: "This information is not available in the internal documents."
- Be concise and precise.
- Answer in the same language as the question.
- Do NOT invent, guess, or assume anything not explicitly stated in the context.

Context from documents:
{context}

Question: {question}

Answer:"""


# ── Chunk quality prompt (used at ingestion time) ─────────────────────────

CHUNK_QUALITY_PROMPT = """You are evaluating the quality of a document chunk for a company knowledge base.

You are given:
1. The filename of the document
2. The text content of the chunk

Your job is to assess TWO things:
A) Is the CONTENT meaningful, readable, real human language? (not gibberish, not random words)
B) Is the DOCUMENT reliable and authoritative based on the filename?

Scoring rules for content quality:
- 0.9–1.0 → clear, structured, informative text (policies, procedures, guidelines)
- 0.6–0.8 → readable but vague or incomplete
- 0.3–0.5 → partial gibberish, very low information density
- 0.0–0.2 → pure gibberish, random words, nonsense, unreadable

Scoring rules for document reliability (filename):
- 1.0 → official, finalized, authoritative (policy, approved, final)
- 0.7–0.9 → mostly reliable
- 0.4–0.6 → draft, unclear status
- 0.1–0.3 → low reliability

CRITICAL rules:
- If filename contains "wip", "draft", "test", "tmp" → doc_score MUST be <= 0.4
- If content contains mostly invented/random words with no real meaning → content_score MUST be <= 0.2
- Random sequences like "velm zortha quillex mortri venzak" are gibberish → content_score: 0.0

Final quality_score = (content_score * 0.7) + (doc_score * 0.3)

Return ONLY valid JSON, no explanation, no markdown:
{{"content_score": <float 0.0-1.0>, "doc_score": <float 0.0-1.0>, "quality_score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Filename: {filename}
Content:
{content}
"""


# ── Agent prompts ─────────────────────────────────────────────────────────

AGENT_PLAN_PROMPT = """You are a search planning agent for a RAG document system.

Your job is to analyze the user question and decide the search strategy.

Rules:
- If the question is SIMPLE (one specific fact) → use 1 query
- If the question is COMPLEX (multi-step, checklist, comparison, summary) → use 2-3 queries
- Queries should be short, keyword-focused, and cover different aspects of the question
- NEVER use filenames as queries, only semantic keywords
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
Queries already tried: {previous_queries}

Rules:
- If you have enough information to answer the question → set "enough" to true
- If important information is clearly missing → set "enough" to false and provide a NEW search query
- The new query MUST be meaningfully different from all queries already tried
- NEVER use filenames as queries, only semantic keywords
- If this is the last iteration → you MUST set "enough" to true regardless
- Return ONLY valid JSON, no explanation, no markdown

Return JSON:
{{
  "enough": true | false,
  "missing": "<what specific information is still missing, or 'nothing' if enough>",
  "next_query": "<focused search query to find missing info, or null if enough>"
}}
"""


# ── Reranking prompt ──────────────────────────────────────────────────────

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
        return RERANKING_PROMPT.replace("<<SOURCE>>", str(source))
    except Exception:
        return f"""
                Return JSON:
                {{"priority": 0.5, "reason": "fallback"}}

                Filename: {source}
                """