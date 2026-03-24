# src/AdpRag/qa.py
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains import RetrievalQA
from .config import OLLAMA_MODEL, TOP_K, MIN_RELEVANCE
from .logger import FileLogger as log

# English prompt for RAG
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

def create_qa_chain(vectorstore):
    """
    Initialize the QA chain using the vectorstore and Ollama LLM.
    """
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    log.info(f"QA chain created with model {OLLAMA_MODEL}")
    return qa_chain