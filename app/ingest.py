import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # <-- NUEVO

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VECTOR_DIR = Path(__file__).resolve().parent.parent / "vectordb"
COLLECTION = "rag_store"

def load_docs(data_dir: Path):
    docs = []
    for p in data_dir.rglob("*"):
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in (".txt", ".md"):
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif p.suffix.lower() == ".docx":
            docs.extend(Docx2txtLoader(str(p)).load())
    return docs

def main():
    load_dotenv(override=True)
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL")  # opcional

    print(f"[INGEST] Cargando documentos de: {DATA_DIR}")
    raw_docs = load_docs(DATA_DIR)
    if not raw_docs:
        print("[INGEST] No se encontraron documentos. Coloca PDFs/TXT/DOCX en ./data")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)
    print(f"[INGEST] Documentos fragmentados en {len(docs)} chunks.")

    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)

    # Chroma persiste automáticamente en persist_directory
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
        collection_name=COLLECTION,
    )
    print(f"[INGEST] Vector DB lista en {VECTOR_DIR} (collection={COLLECTION})")

if __name__ == "__main__":
    main()
