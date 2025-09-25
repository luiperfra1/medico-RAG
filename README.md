# 🩺 Asistente de Documentos Médicos (RAG local con LangChain + Ollama)

Este proyecto es una aplicación **Streamlit** que permite subir documentos médicos en varios formatos (PDF, DOCX, TXT, Markdown) y hacer preguntas en lenguaje natural.  
El sistema está basado en **RAG (Retrieval-Augmented Generation)**: primero busca en tus archivos relevantes y luego genera una respuesta usando un modelo local de **Ollama** (ej. `qwen2.5:3b-instruct`).

---

## ✨ Funcionalidades principales
- 📂 **Subida de documentos médicos** (informes, recetas, resúmenes…).  
- 🔍 **Indexado automático** en una base vectorial local (**ChromaDB**).  
- 💬 **Consultas en lenguaje natural** con respuestas generadas por un LLM local.  
- 🗑️ **Gestión del índice**:  
  - *Reiniciar índice*: borra solo vectores (los archivos permanecen).  
  - *Vaciar todo*: borra vectores + archivos subidos y reinicia la app.  
- 🌐 **Interfaz web sencilla** con Streamlit.  
- ⚙️ **Configuración flexible** de modelo y embeddings mediante `.env`.  

---

## 📌 Limitaciones conocidas
- ✅ Ideal para **preguntas concretas** y localizadas (ej.: *“¿Qué tratamiento aparece para la hipertensión de María López?”*).  
- ⚠️ Menos eficaz en **consultas muy generales o contextos largos**: el sistema fragmenta documentos en *chunks* de ~1000 caracteres y solo pasa unos pocos al modelo (recorte de contexto).  
- ⏳ El rendimiento depende del modelo elegido y de tu hardware:
  - En CPU puede ser lento.  
  - Se recomienda usar GPU o modelos ligeros (`phi3:mini`, `qwen2.5:0.5b`) para mejorar fluidez.  

---

## 🚀 Instalación

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/luiperfra1/medico-RAG.git
   cd medico-RAG
   ```

2. **Crea un entorno virtual**:
   ```bash
   python -m venv .venv
   # Linux / Mac
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Instala dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configura variables de entorno** en un archivo `.env` en la raíz:
   ```env
   OLLAMA_MODEL=qwen2.5:3b-instruct
   EMBED_MODEL=nomic-embed-text
   OLLAMA_BASE_URL=http://localhost:11434  # si usas Ollama local
   ```

5. **Instala Ollama y descarga un modelo**:
   ```bash
   ollama pull qwen2.5:3b-instruct
   ollama pull nomic-embed-text
   ```

---

## ▶️ Uso

1. **Inicia la aplicación**:
   ```bash
   streamlit run app/ui.py
   ```

2. Abre en tu navegador:
   ```
   http://localhost:8501
   ```

3. **Flujo de trabajo**:
   - Arrastra tus documentos (PDF, DOCX, TXT, MD).  
   - Pulsa **Añadir a mi biblioteca** → se indexan automáticamente.  
   - Escribe tu pregunta en **Preguntar a mi biblioteca**.  
   - Pulsa **Buscar respuesta** y verás la respuesta generada a partir de tus documentos.  

4. **Opciones en la barra lateral**:
   - *Reiniciar índice*: limpia solo los vectores.  
   - *Vaciar todo*: borra también los documentos subidos.  

---

## 📂 Estructura del proyecto

```
app/
 ├─ ui.py          # Interfaz Streamlit (subida, preguntas, opciones)
 ├─ rag_chain.py   # Definición de la cadena RAG (prompt, LLM, retriever)
 ├─ ingest.py      # Script CLI para indexar documentos desde /data
uploads/           # Archivos subidos por el usuario (ignorado en git)
vectordb/          # Base vectorial persistente de Chroma (ignorada en git)
data/              # Carpeta opcional para ingesta manual
requirements.txt
.env.example       # Variables de entorno de ejemplo
README.md
```

---

## ⚠️ Notas importantes
- Todo el procesamiento se hace **en local**: tus documentos no se envían a servidores externos.  
- Los modelos grandes de Ollama pueden consumir mucha memoria y tardar en CPU.  
- Para obtener mejores resultados:  
  - Haz preguntas **específicas**.  
  - Sube documentos bien estructurados (informes, recetas, resúmenes claros).  
  - Usa GPU o modelos pequeños si buscas rapidez.  

---

