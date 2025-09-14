"""
pdf_ingest.py
-------------
Handles ingestion of PDFs uploaded through Streamlit.
Steps:
1. Save PDF into Data/processed folder.
2. Call process_pdf() from ingest.py.
3. Return structured JSON outputs for downstream indexing.
"""
import os
from pathlib import Path
from ingest import process_pdf

DATA_DIR = "Data/processed"

def process_uploaded_pdfs(uploaded_files):
    """
    Process PDFs uploaded via Streamlit (UploadedFile objects).
    Saves them into Data/processed and returns JSON outputs.
    """
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    outputs = []

    for file in uploaded_files:
        # Save temporarily
        pdf_path = os.path.join(DATA_DIR, file.name)
        with open(pdf_path, "wb") as f:
            f.write(file.getbuffer())

        # Process with your existing pipeline
        out = process_pdf(pdf_path, DATA_DIR)
        outputs.append(out)

    return outputs
