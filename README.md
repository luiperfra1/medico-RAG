# 🩺 Asistente de Documentos Médicos (RAG local con LangChain + Ollama)

Este proyecto es una aplicación **Streamlit** que permite subir documentos médicos en varios formatos (PDF, DOCX, TXT, Markdown) y hacer preguntas en lenguaje natural.  
El sistema usa **RAG (Retrieval-Augmented Generation)**: busca en tus archivos y genera respuestas usando un modelo local de **Ollama** (ej. `qwen2.5:3b-instruct`).

---

## ✨ Funcionalidades
- Subida de documentos médicos (informes, recetas, resúmenes).  
- Indexado automático en una base vectorial local (**ChromaDB**).  
- Consultas en lenguaje natural con respuestas generadas por LLM local.  
- Opción de **reiniciar índice** (solo borra vectores) o **vaciar todo** (índice + documentos).  
- Interfaz amigable en navegador con Streamlit.  
- Configuración flexible de modelo y embeddings desde `.env`.  

---

## 🚀 Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/luiperfra1/medico-RAG.git
   cd medico-RAG
   ```

2. Crea un entorno virtual:
   ```bash
   python -m venv .venv
   # Linux / Mac
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Crea un archivo `.env` en la raíz con:
   ```env
   OLLAMA_MODEL=qwen2.5:3b-instruct
   EMBED_MODEL=nomic-embed-text
   OLLAMA_BASE_URL=http://localhost:11434  # si usas Ollama local
   ```

5. Asegúrate de tener instalado **Ollama** y haber descargado un modelo:
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

---

## ▶️ Uso

1. Inicia la app:
   ```bash
   streamlit run app/ui.py
   ```

2. Abre en el navegador:
   ```
   http://localhost:8501
   ```

3. Pasos en la interfaz:
   - **Añadir documentos**: arrastra tus PDFs, DOCX, TXT o MD.  
   - Pulsa **Añadir a mi biblioteca** → se indexan automáticamente.  
   - En **Preguntar a mi biblioteca**, escribe tu pregunta.  
   - Pulsa **Buscar respuesta** y obtendrás la información localizada.  

4. Opciones en la barra lateral:
   - **Reiniciar índice** → limpia la base vectorial (los documentos quedan).  
   - **Vaciar todo** → elimina índice + documentos subidos y reinicia la app.  

---

## 📂 Estructura del proyecto

```
app/
 ├─ ui.py          # Interfaz Streamlit
 ├─ rag_chain.py   # Definición de la cadena RAG
uploads/           # Archivos subidos (ignorado en git)
vectordb/          # Base vectorial persistente (ignorada en git)
data/              # Carpeta opcional para ingesta manual
requirements.txt
.env.example       # Variables de entorno de ejemplo
README.md
```

---

## ⚠️ Notas importantes
- Los documentos subidos se procesan **localmente** (no se envían a servicios externos).  
- Rendimiento: los modelos grandes en Ollama pueden tardar bastante en CPU. Se recomienda GPU o modelos pequeños (`phi3:mini`, `qwen2.5:0.5b`).  

---
