# app/rag_chain.py
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# IMPORTANTES: Path, no str
VECTOR_DIR = Path("vectordb")
COLLECTION = "rag_store"

SYSTEM_PROMPT = """Eres un asistente útil. Responde en español, de forma concisa y precisa.
Usa exclusivamente el CONTEXTO cuando sea relevante. Si no hay información suficiente, dilo claramente.
FORMATO:
- Respuesta
- (Si procede) Fuentes: lista breve de títulos/fragmentos
"""

Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Pregunta: {question}\n\nContexto:\n{context}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(f"- {d.metadata.get('source','?')}: {d.page_content}" for d in docs[:6])

def build_chain():
    load_dotenv(override=True)
    base_url = os.getenv("OLLAMA_BASE_URL")  # opcional
    chat_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    llm = ChatOllama(model=chat_model, base_url=base_url, temperature=0.2)

    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | Q_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
