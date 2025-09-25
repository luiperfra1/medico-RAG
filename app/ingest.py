# app/ingest.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# === Rutas y nombres de colección ===
# Usar Path (no str) evita problemas de rutas relativas/absolutas.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"      # Coloca aquí tus PDFs/TXT/DOCX/MD
VECTOR_DIR = Path(__file__).resolve().parent.parent / "vectordb" # Chroma persistirá aquí
COLLECTION = "rag_store"                                         # Nombre lógico de colección


def load_docs(data_dir: Path):
    """
    Carga documentos desde `data_dir` a objetos Document de LangChain.
    Formatos soportados: PDF, DOCX, TXT y MD.
    """
    docs = []
    if not data_dir.exists():
        return docs

    for p in data_dir.rglob("*"):
        suf = p.suffix.lower()
        if suf == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif suf in (".txt", ".md"):
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif suf == ".docx":
            docs.extend(Docx2txtLoader(str(p)).load())
    return docs


def main() -> None:
    """
    Pipeline de ingesta:
      1) Lee variables de entorno (modelo de embeddings y base_url de Ollama).
      2) Carga documentos desde ./data.
      3) Fragmenta en chunks solapados (mejora el recall en QA).
      4) Calcula embeddings y persiste en Chroma (./vectordb).
    """
    load_dotenv(override=True)
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")    # p.ej.: "nomic-embed-text"
    base_url = os.getenv("OLLAMA_BASE_URL")                       # p.ej.: "http://127.0.0.1:11434"

    print(f"[INGEST] Cargando documentos de: {DATA_DIR}")
    raw_docs = load_docs(DATA_DIR)
    if not raw_docs:
        print("[INGEST] No se encontraron documentos. Coloca PDFs/TXT/DOCX/MD en ./data")
        return

    # Heurística típica para RAG: ~1k caracteres con solape ~20%
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)
    print(f"[INGEST] Documentos fragmentados en {len(docs)} chunks.")

    # Embeddings con Ollama (modelo configurable por env var)
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)

    # Chroma crea (o amplía) la colección y persiste automáticamente en `persist_directory`
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
        collection_name=COLLECTION,
    )
    print(f"[INGEST] Vector DB lista en {VECTOR_DIR} (collection={COLLECTION})")


if __name__ == "__main__":
    main()
