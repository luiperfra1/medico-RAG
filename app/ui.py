# app/ui.py
from __future__ import annotations

import gc
import os
import shutil
import stat
import time
import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from rag_chain import build_chain, VECTOR_DIR, COLLECTION  # VECTOR_DIR debe ser Path

# === Rutas base de la app ===
ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"   # Archivos subidos por el usuario (persisten entre sesiones)
DATA_DIR = ROOT / "data"        # Carpeta opcional para arrastrar ficheros fuera de la UI

st.set_page_config(page_title="Asistente de Documentos Médicos", layout="wide")


# -------------------------------------------------------------------------
# Utilidades de sistema y limpieza (pensadas para Windows y ficheros bloqueados)
# -------------------------------------------------------------------------
def _onerror_make_writable(func, path, exc_info):
    """
    Si durante rmtree encontramos un archivo/carpeta de solo lectura (típico en Windows),
    quitamos el flag de solo lectura y reintentamos la operación.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def _try_delete_dir(dirpath: Path, attempts: int = 6, wait_s: float = 0.35) -> bool:
    """
    Intenta borrar una carpeta con reintentos para evitar 'PermissionError' por locks temporales.
    Devuelve True si se logró borrar; False en caso contrario.
    """
    for _ in range(attempts):
        try:
            shutil.rmtree(dirpath, onerror=_onerror_make_writable)
            return True
        except PermissionError:
            time.sleep(wait_s)
            gc.collect()
        except FileNotFoundError:
            return True
        except Exception:
            time.sleep(wait_s)
            gc.collect()
    return False


# -------------------------------------------------------------------------
# Ingesta y carga de documentos
# -------------------------------------------------------------------------
def _save_uploads(uploaded_files: Iterable[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    """
    Guarda los ficheros subidos por el usuario en UPLOAD_DIR.
    Mantiene el nombre original (espacios -> guiones bajos).
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for uf in uploaded_files:
        fname = uf.name.replace(" ", "_")
        dest = UPLOAD_DIR / fname
        with open(dest, "wb") as f:
            f.write(uf.read())
        saved.append(dest)
    return saved


def _load_docs_from_paths(paths: Iterable[Path]):
    """
    Carga documentos a objetos LangChain Document, soportando PDF, DOCX, TXT y MD.
    """
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


def _ingest_paths(paths: Iterable[Path]) -> int:
    """
    Ingesta incremental en la colección Chroma (crea si no existe).
    1) Carga documentos
    2) Fragmenta en chunks solapados (mejora el recall)
    3) Embebe y añade a la colección
    Devuelve el nº de fragments añadidos.
    """
    load_dotenv(override=True)
    base_url = os.getenv("OLLAMA_BASE_URL")                 # p.ej. "http://127.0.0.1:11434"
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    raw_docs = _load_docs_from_paths(paths)
    if not raw_docs:
        return 0

    # Heurística saludable para QA: chunks ~1k tokens con 20% solape
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


# -------------------------------------------------------------------------
# Acciones de mantenimiento del índice y del estado de la app
# -------------------------------------------------------------------------
def _drop_chroma_collection() -> None:
    try:
        load_dotenv(override=True)
        base_url = os.getenv("OLLAMA_BASE_URL")
        embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

        embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
        vectordb = Chroma(
            persist_directory=str(VECTOR_DIR),
            collection_name=COLLECTION,
            embedding_function=embeddings,
        )
        # Vaciar documentos por si delete_collection no está disponible en tu versión
        try:
            vectordb._collection.delete(where={})
        except Exception:
            pass
        # Borrar la colección (si está soportado)
        try:
            vectordb.delete_collection()
        except Exception:
            pass
    except Exception:
        pass

def _index_count() -> int:
    """
    Devuelve el número de embeddings en la colección persistida.
    Si no existe, devuelve 0.
    """
    try:
        load_dotenv(override=True)
        base_url = os.getenv("OLLAMA_BASE_URL")
        embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

        embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
        vectordb = Chroma(
            persist_directory=str(VECTOR_DIR),
            collection_name=COLLECTION,
            embedding_function=embeddings,
        )
        # API interna pero fiable para contar
        return int(vectordb._collection.count())
    except Exception:
        return 0

def _reset_vectordb_soft() -> Tuple[bool, str]:
    """
    Reinicia el índice vectorial (solo vectores):
      - Suelta la colección Chroma (libera locks de SQLite).
      - Intenta borrar la carpeta del índice con reintentos.
      - Si falla, la renombra para borrado diferido en el próximo arranque.
      - Invalida caché y fuerza recarga de la app.
    """
    try:
        # 0) Limpiar cachés y soltar colección para evitar locks
        st.cache_resource.clear()
        gc.collect()
        time.sleep(0.25)
        _drop_chroma_collection()

        vdir = Path(VECTOR_DIR)
        if vdir.exists():
            # 1) Intentar borrado con reintentos
            if not _try_delete_dir(vdir):
                # 2) Plan B: renombrar y borrar más tarde
                tomb = vdir.with_name(vdir.name + f"_to_delete_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                try:
                    os.replace(vdir, tomb)
                    st.session_state["pending_delete"] = str(tomb)
                except Exception as e:
                    return False, f"No se pudo reiniciar (borrar/renombrar): {e}"

        # 3) Invalidar caché de la cadena y refrescar UI
        st.session_state["index_version"] = int(time.time())
        st.rerun()  # fuerza recarga para que _get_chain() se reconstruya

        return True, "Índice reiniciado."
    except Exception as e:
        return False, f"No se pudo reiniciar: {e}"



def _wipe_all() -> Tuple[bool, str]:
    """
    Borra índice + archivos subidos y reinicia la app (con fallback robusto en Windows).
    """
    try:
        # Cerrar caches para soltar locks del índice
        st.cache_resource.clear()
        gc.collect()
        time.sleep(0.25)

        # 0) Soltar/limpiar la colección Chroma en disco (evita locks de SQLite)
        _drop_chroma_collection()

        # 1) Borrar (o renombrar) índice
        vdir = Path(VECTOR_DIR)
        if vdir.exists():
            if not _try_delete_dir(vdir):
                tomb = vdir.with_name(vdir.name + f"_trash_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                try:
                    os.replace(vdir, tomb)   # mueve la carpeta aunque haya ficheros bloqueados
                    st.session_state["pending_delete"] = str(tomb)  # se intentará borrar al arrancar
                except Exception as e:
                    return False, f"No se pudo borrar/renombrar el índice: {e}"

        # 2) Borrar subidas
        if UPLOAD_DIR.exists() and not _try_delete_dir(UPLOAD_DIR):
            return False, "No se pudieron borrar las subidas."

        # 3) Limpiar estado y forzar nueva clave de cache
        st.session_state.clear()
        st.session_state["index_version"] = int(time.time())  # clave nueva para invalidar _get_chain
        st.session_state["uploader_key"] = f"uploader_{time.time()}"

        st.rerun()  # reinicia la app
        return True, "Todo vaciado correctamente."
    except Exception as e:
        return False, f"No se pudo vaciar todo: {e}"




# -------------------------------------------------------------------------
# Cadena RAG (cacheada) y utilidades de versión de índice
# -------------------------------------------------------------------------
@st.cache_resource
def _get_chain(version: int):
    """
    Devuelve la cadena RAG lista para invocar. Se cachea por versión de índice;
    al cambiar la versión forzamos recarga para que apunte al nuevo Chroma.
    """
    return build_chain()


def _bump_index_version() -> None:
    """
    Incrementa el contador de versión del índice; provoca que _get_chain()
    recargue la cadena con el nuevo estado del vector store.
    """
    st.session_state["index_version"] = st.session_state.get("index_version", 0) + 1


# -------------------------------------------------------------------------
# UI principal
# -------------------------------------------------------------------------
def main() -> None:
    # Si hay una carpeta “pendiente de borrar” de un reinicio previo, intentamos eliminarla ahora
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
        "Sube informes, recetas o resúmenes y realiza preguntas en lenguaje natural. "
        "El sistema responderá apoyándose en tus documentos."
    )

    # (1) Subir documentos
    st.markdown("### 1) Añadir documentos")
    st.caption("Formatos: PDF, DOCX, TXT y Markdown.")
    uploaded = st.file_uploader(
        "Arrastra aquí tus documentos o pulsa en “Examinar archivos”.",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=st.session_state.get("uploader_key", "uploader_0"),
    )

    add_clicked = st.button("Añadir a mi biblioteca", use_container_width=True)

    if add_clicked:
        if not uploaded:
            st.warning("Primero selecciona al menos un archivo.")
        else:
            saved = _save_uploads(uploaded)
            with st.spinner("Procesando e indexando tus documentos..."):
                n_chunks = _ingest_paths(saved)
            _bump_index_version()
            st.success(f"Documentos añadidos. Fragmentos indexados: {n_chunks}")

    st.divider()

    # (2) Consultar
    st.markdown("### 2) Preguntar a mi biblioteca")
    st.caption('Ejemplo: "¿Qué tratamiento aparece para la hipertensión de María López?"')

    chain = _get_chain(st.session_state["index_version"])
    q = st.text_input("Escribe tu pregunta")
    ask = st.button("Buscar respuesta", use_container_width=True)

    if ask and q.strip():
        n = _index_count()
        st.subheader("Respuesta")
        if n == 0:
            st.write("No hay documentos indexados ahora mismo. Sube archivos y vuelve a intentarlo.")
        else:
            with st.spinner("Buscando en tus documentos..."):
                answer = chain.invoke(q.strip())
            st.write(answer)



    # Lateral: estado y mantenimiento
    with st.sidebar:
        st.markdown("#### Mi biblioteca")
        total_uploads = sum(1 for _ in UPLOAD_DIR.glob("*")) if UPLOAD_DIR.exists() else 0
        st.write(f"Archivos en ‘subidas’: {total_uploads}")
        st.write(f"Fragmentos indexados: {_index_count()}")

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
                "- **Reiniciar índice**: borra únicamente la base vectorial interna.\n"
                "- **Vaciar todo**: borra también los archivos que has subido en esta app."
            )


if __name__ == "__main__":
    main()
