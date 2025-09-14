"""
index_documents.py
------------------
Indexes extracted JSON files into ChromaDB with local embeddings.
- Loads JSONs from Data/processed
- Splits text into chunks
- Adds chunks to ChromaDB
"""
import os
import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_DIR
from chromadb.api.types import EmbeddingFunction

# -------------------------------
# Setup Local Embeddings (correct Chroma interface)
# -------------------------------
class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()

    def name(self):
        return "local-sentence-transformer"

embedding_fn = LocalEmbeddingFunction()

# -------------------------------
# Setup ChromaDB
# -------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn
)

# -------------------------------
# JSON Loader
# -------------------------------
def load_extracted_json(json_folder):
    docs = []
    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"❌ Folder not found: {json_folder}")
    
    for fname in os.listdir(json_folder):
        if fname.endswith(".json"):
            with open(os.path.join(json_folder, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                docs.append({"id": fname.replace(".json",""), "content": data})
    return docs

# -------------------------------
# Text Chunker
# -------------------------------
def chunk_text(text, max_tokens=2000):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_tokens:
            current += sent + ". "
        else:
            chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# -------------------------------
# Index Documents
# -------------------------------
def index_documents(json_folder):
    docs = load_extracted_json(json_folder)
    for doc in docs:
        doc_id = doc["id"]
        content = doc["content"]

        if isinstance(content, dict):
            text = json.dumps(content)
        else:
            text = str(content)

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"

            existing = collection.get(ids=[chunk_id])
            if existing and existing.get("ids"):
                print(f"⏭ Skipping {chunk_id} (already indexed)")
                continue

            collection.add(
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[{"source": doc_id}]
            )
            print(f" Indexed {chunk_id}")

    print("\n🎉 All documents indexed successfully (Local Embeddings)!")

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        default="Data/processed",
        help="Folder containing processed JSON files"
    )
    args = parser.parse_args()

    index_documents(json_folder=args.json_dir)
