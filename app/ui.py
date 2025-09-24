# app/ui.py
from __future__ import annotations
import os
import shutil
import gc
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from rag_chain import build_chain, VECTOR_DIR, COLLECTION  # VECTOR_DIR debe ser Path en rag_chain.py
import os, stat, datetime

def _onerror_make_writable(func, path, exc_info):
    """Quita solo-lectura y reintenta (útil en Windows)."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass

def _try_delete_dir(dirpath: Path, attempts: int = 6, wait_s: float = 0.35) -> bool:
    """Intenta borrar una carpeta con reintentos."""
    for _ in range(attempts):
        try:
            shutil.rmtree(dirpath, onerror=_onerror_make_writable)
            return True
        except PermissionError:
            time.sleep(wait_s)
            gc.collect()
    return False

# Carpetas
ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
DATA_DIR = ROOT / "data"  # sigue disponible si quieres arrastrar aquí ficheros fuera de la UI

st.set_page_config(page_title="Asistente de Documentos Médicos", layout="wide")

# --------------------------
# Utilidades de carga/ingesta
# --------------------------
def _save_uploads(uploaded_files):
    """Guarda los archivos subidos en /uploads."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for uf in uploaded_files:
        fname = uf.name.replace(" ", "_")
        dest = UPLOAD_DIR / fname
        with open(dest, "wb") as f:
            f.write(uf.read())
        saved.append(dest)
    return saved

def _load_docs_from_paths(paths):
    docs = []
    for p in paths:
        suf = p.suffix.lower()
        if suf == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif suf in (".txt", ".md"):
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif suf == ".docx":
            docs.extend(Docx2txtLoader(str(p)).load())
    return docs
def _try_delete_dir(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath, ignore_errors=True)

def _wipe_all():
    """Borra índice, archivos subidos y reinicia la app."""
    try:
        # cerrar cachés para soltar locks del índice
        st.cache_resource.clear()
        gc.collect()
        time.sleep(0.25)

        # borrar índice
        vdir = Path(VECTOR_DIR)
        if vdir.exists():
            shutil.rmtree(vdir, ignore_errors=True)

        # borrar subidas
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

        # limpiar todo el estado de sesión
        st.session_state.clear()

        # reiniciar la app (esto resetea también el uploader)
        st.session_state["uploader_key"] = f"uploader_{time.time()}"

        st.rerun()
    except Exception as e:
        st.warning(f"No se pudo vaciar todo: {e}")


def _ingest_paths(paths):
    """Ingesta incremental a la colección (crea si no existe)."""
    load_dotenv(override=True)
    base_url = os.getenv("OLLAMA_BASE_URL")  # opcional
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    raw_docs = _load_docs_from_paths(paths)
    if not raw_docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
    vectordb.add_documents(docs)
    return len(docs)

def _reset_vectordb_soft():
    """
    Reinicio robusto para Windows:
    - Limpia la cache (cierra conexiones)
    - Intenta borrar con reintentos
    - Si no puede, renombra y programa el borrado tras reiniciar
    """
    try:
        # 1) Cerrar recursos/cachés
        st.cache_resource.clear()
        gc.collect()
        time.sleep(0.25)

        vdir = Path(VECTOR_DIR)  # por si llega str
        if not vdir.exists():
            return True, "Índice reiniciado."

        # 2) Reintentos de borrado
        if _try_delete_dir(vdir):
            return True, "Índice reiniciado."

        # 3) Plan B: renombrar y borrar luego
        tomb = vdir.with_name(vdir.name + f"_to_delete_{datetime.datetime.now().strftime('%H%M%S')}")
        try:
            os.replace(vdir, tomb)   # mueve la carpeta aunque haya ficheros bloqueados
            # guardamos para intentar borrarla al siguiente arranque
            st.session_state["pending_delete"] = str(tomb)
            return True, "Índice reiniciado (se terminará de limpiar al reiniciar la app)."
        except Exception as e:
            return False, f"No se pudo reiniciar: {e}"

    except Exception as e:
        return False, f"No se pudo reiniciar: {e}"


# --------------------------
# Cache de la cadena RAG
# --------------------------
@st.cache_resource
def _get_chain(version: int):
    return build_chain()

def _bump_index_version():
    st.session_state["index_version"] = st.session_state.get("index_version", 0) + 1

# --------------------------
# UI
# --------------------------
def main():
    pending = st.session_state.get("pending_delete")
    if pending:
        p = Path(pending)
        if p.exists():
            _try_delete_dir(p, attempts=4, wait_s=0.25)
        st.session_state.pop("pending_delete", None)
    load_dotenv(override=True)
    if "index_version" not in st.session_state:
        st.session_state["index_version"] = 0

    # Encabezado
    st.title("Asistente de Documentos Médicos")
    st.write(
        "Sube informes, recetas o resúmenes. "
        "Haz preguntas en lenguaje natural y obtén respuestas basadas en tus documentos."
    )

    # Paso 1: Subir documentos
    st.markdown("### 1) Añadir documentos")
    st.caption("Formatos permitidos: PDF, DOCX, TXT y Markdown.")
    uploaded = st.file_uploader(
        "Arrastra aquí tus documentos o pulsa en “Examinar archivos”.",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=st.session_state.get("uploader_key", "uploader_0")
    )

    add_cols = st.columns([1])
    with add_cols[0]:
        add_clicked = st.button("Añadir a mi biblioteca", use_container_width=True)

    if add_clicked:
        if not uploaded:
            st.warning("Primero selecciona al menos un archivo.")
        else:
            saved = _save_uploads(uploaded)
            with st.spinner("Procesando e indexando tus documentos..."):
                n_chunks = _ingest_paths(saved)
            _bump_index_version()
            st.success(f"Documentos añadidos correctamente. Fragmentos indexados: {n_chunks}")

    st.divider()

    # Paso 2: Consultar
    st.markdown("### 2) Preguntar a mi biblioteca")
    st.caption('Ejemplo: "¿Qué tratamiento aparece para la hipertensión de María López?"')

    chain = _get_chain(st.session_state["index_version"])
    q = st.text_input("Escribe tu pregunta")
    ask = st.button("Buscar respuesta", use_container_width=True)

    if ask and q.strip():
        with st.spinner("Buscando en tus documentos..."):
            answer = chain.invoke(q.strip())
        st.subheader("Respuesta")
        st.write(answer)

    # Lateral: estado simple y opciones
    with st.sidebar:
        st.markdown("#### Mi biblioteca")
        total_uploads = sum(1 for _ in UPLOAD_DIR.glob("*")) if UPLOAD_DIR.exists() else 0
        st.write(f"Archivos en ‘subidas’: {total_uploads}")

        st.markdown("#### Opciones")
        if st.button("Reiniciar índice (solo vectores)"):
            ok, msg = _reset_vectordb_soft()
            if ok:
                _bump_index_version()
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

        if st.button("Vaciar todo (archivos + índice)"):
            ok, msg = _wipe_all()
            if ok:
                _bump_index_version()
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

        with st.expander("Ayuda"):
            st.write(
                "- **Reiniciar índice** borra únicamente la base interna de búsqueda.\n"
                "- **Vaciar todo** borra también los archivos que has subido en esta app."
            )


if __name__ == "__main__":
    main()
