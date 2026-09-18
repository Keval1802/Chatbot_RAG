# ChatGroq Document RAG

A Document Retrieval-Augmented Generation (RAG) system using LangChain, HuggingFace Embeddings, FAISS vector store, and Groq LLM.

## UI Entry Points

This repository provides two user interfaces:

- `app.py` - Streamlit application
- `hf_app.py` - Gradio application

Both interfaces utilize the core backend logic in `rag_backend.py`.

## Architecture

```text
PDF/TXT upload
      ↓
LangChain loaders
      ↓
RecursiveCharacterTextSplitter
      ↓
HuggingFace all-MiniLM-L6-v2 embeddings
      ↓
FAISS Vector Store
      ↓
History-aware retrieval
      ↓
ChatGroq LLM
      ↓
Grounded answer
```

## Embeddings

This project uses `sentence-transformers/all-MiniLM-L6-v2` for generating document vector embeddings.

## Project Structure

```text
app.py
hf_app.py
rag_backend.py
requirements.txt
README.md
.env.example
.gitignore
.streamlit/config.toml
```

## Setup & Running Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Set your Groq API key:

Windows CMD:
```cmd
set "GROQ_API_KEY=YOUR_REAL_GROQ_API_KEY"
```

PowerShell:
```powershell
$env:GROQ_API_KEY="YOUR_REAL_GROQ_API_KEY"
```

### 3. Run Application

For Streamlit interface:
```bash
streamlit run app.py
```

For Gradio interface:
```bash
python hf_app.py
```

## Environment & Secrets

Never commit sensitive API keys or secret files to the repository:

```text
.env
.streamlit/secrets.toml
GROQ_API_KEY
```
