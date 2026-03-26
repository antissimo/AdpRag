from .logger import FileLogger as log

class QueryTransformer:
    """
    Transforms user questions into optimized search queries
    for retrieval in RAG pipelines.
    """

    _instance = None

    @classmethod
    def get(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, llm=None):
        if llm is None:
            from .llm import RAGLLM
            self.llm = RAGLLM.get()
        else:
            self.llm = llm

    def transform(self, question: str) -> str:
        """
        Transform a single question into a concise, retrieval-friendly query.
        """
        try:
            question = question.strip()
            if not question:
                log.warning("Empty question passed to QueryTransformer, returning fallback.")
                return question

            prompt = f"""
                Rewrite the user question into an optimized search query.
                - Use keywords
                - Be concise
                - Remove unnecessary words
                - Keep meaning intact

                Original question: {question}

                Search query:
                """
            log.info(f"Transforming query: {question}")
            response = self.llm.invoke(prompt)
            transformed_query = response.strip()
            log.info(f"Transformed query: {transformed_query}")
            return transformed_query

        except Exception as e:
            log.warning(f"Query transformation failed: {e}. Returning original question.")
            return question

    def transform_multi(self, question: str, n: int = 3) -> list[str]:
        """
        Generate multiple variant queries for better retrieval.
        """
        try:
            question = question.strip()
            if not question:
                return [question]

            prompt = f"""
                Generate {n} different search queries for the following question.
                Each query should be concise and keyword-focused.
                Return each query on a new line.

                Question: {question}
                """
            response = self.llm.invoke(prompt)
            queries = [q.strip() for q in response.split("\n") if q.strip()]
            if not queries:
                return [question]

            return queries[:n]

        except Exception as e:
            log.warning(f"Multi-query transformation failed: {e}. Returning original question.")
            return [question]