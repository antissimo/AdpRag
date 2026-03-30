# src/AdpRag/qa.py

from langchain_core.prompts import PromptTemplate
from .llm import RAGLLM
from .logger import FileLogger as log
from .instructions import PROMPT_TEMPLATE


class SimpleQAChain:
    def __init__(self):
        self.llm = RAGLLM.get()
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        log.info("Simple QA chain initialized")

    def invoke(self, question: str, docs: list):
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)

        return {
            "result":           response.strip(),
            "source_documents": docs,
        }


def create_qa_chain(vectorstore=None):
    return SimpleQAChain()