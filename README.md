
# Document Q&A Agent — xChat Prototype

**One-line:** A retrieval-augmented Document Q&A agent that ingests PDFs, extracts structured content, indexes chunks into a vector DB (Chroma), and answers natural-language queries using an LLM (Gemini). Includes a CLI, an interactive Streamlit UI, and an ArXiv search helper.

This README explains step-by-step how to set up, run, test, and package the project for submission. Follow the steps exactly.

---

# Table of contents
- [What this repo contains](#what-this-repo-contains)
- [Prerequisites](#prerequisites)
- [Install & Setup](#install--setup)
- [Environment variables / config](#environment-variables--config)
- [Folder layout (important files)](#folder-layout-important-files)
- [Workflow overview](#workflow-overview)
- [Step-by-step usage (commands)](#step-by-step-usage-commands)
  - [1) Ingest PDFs (batch)](#1-ingest-pdfs-batch)
  - [2) Index processed JSON into ChromaDB](#2-index-processed-json-into-chromadb)
  - [3) Query — quick tests & CLI RAG Chatbot](#3-query---quick-tests--cli-rag-chatbot)
  - [4) Streamlit web UI (upload + chat)](#4-streamlit-web-ui-upload--chat)
  - [5) ArXiv search (bonus, free)](#5-arxiv-search-bonus-free)
- [Automated tests / test queries](#automated-tests--test-queries)
- [Troubleshooting & common errors](#troubleshooting--common-errors)
- [Production notes (enterprise readiness)](#production-notes-enterprise-readiness)
- [Packaging & submission checklist](#packaging--submission-checklist)
- [Demo video script (ready to record)](#demo-video-script-ready-to-record)
- [FAQ & final notes](#faq--final-notes)
- [License](#license)

---

# What this repo contains

Key capabilities implemented:

- PDF ingestion and structured extraction (titles, sections, tables, images, references) — `src/ingest.py`  
- Streamlit-friendly upload wrapper for ingestion — `src/pdf_ingest.py`  
- Document chunking + indexing into ChromaDB with local SentenceTransformer embeddings — `src/index_documents.py`  
- Retrieval + prompt building + Gemini call (RAG) — `src/rag_agent.py`  
- CLI chat demo + ArXiv helper — `src/chat_rag.py`, `src/arxiv_helper.py`  
- Streamlit UI for upload + chat — `src/app_streamlit.py`  
- Helper test runner — `src/test_queries.py` (if present)  
- Configuration loader — `src/config.py` (reads .env)

---

# Prerequisites

- Python 3.10+ recommended.
- ~8 GB free disk for embeddings/cache. Use more for many PDFs.
- Internet for LLM API (Gemini) & ArXiv search.
- (Optional) GPU for fast embedding if using GPU-enabled SentenceTransformer, else CPU fine.
- Windows / macOS / Linux — commands below include both PowerShell and POSIX variants.

---

# Install & Setup

1. Clone the repository (or unzip your project folder):


git clone https://github.com/<your-username>/<repo>.git
cd <repo>


2. Create and activate a virtual environment (Windows PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or (macOS / Linux / WSL):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If you need a minimal `requirements.txt`, this project uses:
> ```
> pymupdf
> pdfplumber
> python-dotenv
> google-generativeai
> chromadb
> sentence-transformers
> pandas
> tiktoken
> streamlit
> arxiv
> colorama
> fastapi
> uvicorn
> ```
> (You already have `requirements.txt` — `pip install -r requirements.txt` is recommended.)

---

# Environment variables / config

Create a `.env` file in the project root (do **not** commit this). Example `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_DB_DIR=chroma_store    # path where Chroma will persist DB
```

`src/config.py` should load these values.

---

# Folder layout (important files)

```
.
├─ Data/
│  ├─ pdfs/            # put your raw PDF files here for batch ingest
│  └─ processed/       # generated JSON and extracted assets (tables/images)
├─ src/
│  ├─ ingest.py        # full PDF structured extractor (batch)
│  ├─ pdf_ingest.py    # wrapper to process Streamlit uploaded files
│  ├─ index_documents.py # chunker + local embeddings -> Chroma
│  ├─ rag_agent.py     # retrieval, prompt building, Gemini call
│  ├─ query_documents.py # retrieval-only test script
│  ├─ chat_rag.py      # CLI interactive chatbot (uses rag_agent)
│  ├─ app_streamlit.py # Streamlit UI (upload + query)
│  └─ arxiv_helper.py  # arXiv search helper
├─ chroma_store/       # persistent Chroma DB (defined by CHROMA_DB_DIR)
├─ requirements.txt
├─ .env                # (local, NOT committed)
└─ README.md
```

---

# Workflow overview

1. **Prepare PDFs** (put them into `Data/pdfs/` or upload via Streamlit UI).
2. **Extract structured JSON** using `ingest.py` or Streamlit upload (`process_uploaded_pdfs`).
3. **Index JSON** into ChromaDB with `index_documents.py` (creates embeddings & stores them).
4. **Query** via:
   - CLI: `python src/chat_rag.py`
   - RAG single query: `python src/rag_agent.py --query "..."`  
   - Streamlit UI: `streamlit run src/app_streamlit.py`
5. (Optional) Search ArXiv via `app_streamlit.py` or `chat_rag.py`.

---

# Step-by-step usage (commands)


Step-by-step usage (commands)

All commands assume your virtualenv .venv is activated.

1) Ingest PDFs (batch)

Place PDFs in Data/pdfs/ then run:

python src/ingest.py --input_dir Data/pdfs --output_dir Data/processed


or (POSIX):

python src/ingest.py --input_dir Data/pdfs --output_dir Data/processed


What happens:

ingest.py processes each PDF and writes a JSON for each file to Data/processed/.

It extracts pages, blocks, headings, tables (CSV in Data/processed/tables/), images (in Data/processed/images/) and writes documents_manifest.json.

If you upload via Streamlit, use Process PDFs button in the UI which calls process_uploaded_pdfs and saves JSON files under Data/processed/.

2) Index processed JSON into ChromaDB

Run the indexer (this creates or updates your persistent Chroma DB at CHROMA_DB_DIR):

python src/index_documents.py --json_dir Data/processed


What this does:

Loads all JSON files from Data/processed.

Chunks text (configurable).

Creates embeddings (local SentenceTransformer by default) and inserts chunks into the documents Chroma collection.

Notes:

If you want to use Gemini (cloud) embeddings instead, modify index_documents.py to use the Google embedding function or your preferred embedding provider. Local embeddings are recommended to avoid API quotas.

3) Query — quick tests & CLI RAG Chatbot
Retrieval-only test (inspect top chunks):
python src/query_documents.py --query "What is the main idea of transformers?" --top_k 5


This prints top-k retrieved chunks and their source filenames (no LLM call).

One-shot RAG answer:
python src/rag_agent.py --query "What is the main contribution of the Transformer model?" --top_k 6


This will:

Retrieve top-k chunks from Chroma.

Build the prompt with contexts and instructions.

Call Gemini (requires GEMINI_API_KEY).

Print the answer and the sources.

CLI interactive chat:
python src/chat_rag.py --top_k 5


Type questions interactively. To search ArXiv from CLI, type phrases like find paper on quantum error correction John Preskill or include arxiv in the query.

4) Streamlit web UI (upload + chat)

Start the Streamlit app:

streamlit run src/app_streamlit.py


Flow in UI:

Upload PDFs in sidebar.

Click Process PDFs — this runs process_uploaded_pdfs() (saves JSON to Data/processed) and then index_documents() (adds to Chroma).

Go to Ask Questions, type your query, set top_k, click Get Answer.

Important: If you processed PDFs previously from CLI, the Streamlit app can still query the same Chroma store because index_documents.py writes to CHROMA_DB_DIR.

5) ArXiv search (bonus, free)

From Streamlit UI or CLI (chat_rag.py), include queries with find paper or arxiv. This uses arxiv Python package (no LLM credits used).

Example:

find paper on quantum error correction by John Preskill





# Packaging & submission checklist

- [ ] All code pushed to a GitHub repo (or zipped).  
- [ ] `.gitignore` present (excludes `.env`, `.venv`, `Data/processed` optionally).  
- [ ] README.md (this file) in project root.  
- [ ] `requirements.txt` up-to-date (pin versions if possible).  
- [ ] Demo video recorded and saved as `demo.mp4`.  
- [ ] Quick smoke test passes.

---




