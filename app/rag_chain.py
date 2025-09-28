# app/rag_chain.py
from __future__ import annotations

from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === Configuración base del vector store ===
# Usar Path (no str) evita ambigüedades con rutas relativas/absolutas
VECTOR_DIR = Path(__file__).resolve().parent.parent / "vectordb"   # <-- antes: Path("vectordb")
COLLECTION = "rag_store"

# === Prompt del sistema ===
# Instrucciones cortas y claras para un estilo consistente y controlado.
SYSTEM_PROMPT = """Eres un asistente útil. Responde en español, de forma concisa y precisa.
Usa exclusivamente el CONTEXTO cuando sea relevante. Si no hay información suficiente, dilo claramente.
FORMATO:
- Respuesta
- (Si procede) Fuentes: lista breve de títulos/fragmentos
"""

# Plantilla final: el usuario aporta {question} y el retriever aporta {context}
Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Pregunta: {question}\n\nContexto:\n{context}"),
    ]
)


def format_docs(docs) -> str:
    """
    Convierte los documentos recuperados en un bloque de texto legible.
    - Limita a los 6 primeros para evitar prompt demasiado largo.
    - Incluye 'source' si está disponible para trazabilidad.
    """
    return "\n\n".join(
        f"- {d.metadata.get('source', '?')}: {d.page_content}"
        for d in docs
    )


def build_chain(k: int = 10, temperature: float = 0.2):
    """
    Construye la cadena RAG:
      1) Carga configuración desde variables de entorno (opcional).
      2) Crea el LLM de chat en Ollama.
      3) Inicializa embeddings + Chroma y su retriever.
      4) Encadena: {context, question} -> PROMPT -> LLM -> texto (StrOutputParser).

    Parámetros:
      - k: nº de fragmentos recuperados (trade-off recall/latencia).
      - temperature: control de aleatoriedad del LLM (0.0–0.7 recomendado para QA).

    Variables de entorno soportadas:
      - OLLAMA_BASE_URL  (p.ej., "http://127.0.0.1:11434")
      - OLLAMA_MODEL     (p.ej., "qwen2.5:3b-instruct", "llama3:instruct", etc.)
      - EMBED_MODEL      (p.ej., "nomic-embed-text")
    """
    load_dotenv(override=True)

    base_url = os.getenv("OLLAMA_BASE_URL")                  # Puede ser None si usa el default local
    chat_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # LLM de conversación (temperatura baja para respuestas deterministas en QA)
    llm = ChatOllama(model=chat_model, base_url=base_url, temperature=temperature)

    # Embeddings + almacén vectorial persistente
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    # Recuperación de pasajes: k controla el nº de contextos que llegan al prompt
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # Cadena: prepara inputs, aplica prompt, llama al LLM y parsea a str
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | Q_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
