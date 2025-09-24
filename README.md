# RAG con LangChain + Ollama (local)

## 1) Requisitos

- Python 3.10+
- Ollama instalado y ejecutándose (`ollama serve`)
- Modelos:
  - LLM: `ollama pull qwen2.5:3b-instruct` (o `llama2:latest`)
  - Embeddings: `ollama pull nomic-embed-text` (o `mxbai-embed-large`)

## 2) Instalación

```bash
python -m venv .venv
# En PowerShell (Windows):
.venv\Scripts\Activate.ps1
# Si PS dice que no puedes ejecutar scripts:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
cp .env.example .env
