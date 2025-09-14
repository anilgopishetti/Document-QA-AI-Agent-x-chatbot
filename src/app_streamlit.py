# src/app_streamlit.py
import streamlit as st
import os
from pdf_ingest import process_uploaded_pdfs
from index_documents import index_documents
from rag_agent import rag_answer

DATA_DIR = "Data/processed"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="PDF RAG App", page_icon="📚", layout="wide")
st.title(" Research PDF RAG Assistant")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", [" Upload PDFs", " Ask Questions"])

# -------------------------------
# Upload + Ingest PDFs
# -------------------------------
if page == " Upload PDFs":
    st.subheader("Upload and Process PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            outputs = process_uploaded_pdfs(uploaded_files)

            # Index the processed JSONs automatically
            index_documents(DATA_DIR)

        st.success(" PDFs processed and indexed successfully!")
        st.json(outputs)

# -------------------------------
# Query Interface
# -------------------------------
elif page == " Ask Questions":
    st.subheader("Query Indexed Documents")
    query = st.text_input("Enter your research question:")

    top_k = st.slider("Number of contexts to retrieve", 2, 10, 6)

    if st.button("Get Answer") and query.strip():
        with st.spinner("Retrieving answer from RAG..."):
            result = rag_answer(query, top_k=top_k)

        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("###  Answer")
            st.write(result["answer"])

            st.markdown("###  Sources")
            for s in result["sources"]:
                st.write(f"- {s}")

            st.caption(f" Contexts used: {result['num_contexts']}")
