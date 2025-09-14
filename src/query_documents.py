"""
Simple retrieval script that queries Chroma and prints retrieved chunks + metadata.
Usage:
  python src/query_documents.py --query "What is self-attention?" --top_k 5
"""

import argparse
from config import CHROMA_DB_DIR
import chromadb

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(name="documents")

def retrieve(query: str, top_k: int = 5):
    collection = get_collection()
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    # results is a dict with keys: 'ids', 'documents', 'metadatas', 'distances' (structure depends on SDK)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    out = []
    for i, (did, doc, meta, dist) in enumerate(zip(ids, docs, metadatas, distances)):
        out.append({
            "rank": i+1,
            "id": did,
            "source": meta.get("source") if isinstance(meta, dict) else meta,
            "distance": dist,
            "snippet": (doc[:600] + "...") if len(doc) > 600 else doc
        })
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    results = retrieve(args.query, args.top_k)
    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} results for: {args.query}\n")
    for r in results:
        print(f"[{r['rank']}] id={r['id']} source={r['source']} distance={r['distance']:.4f}")
        print(r['snippet'])
        print("-" * 80)

if __name__ == "__main__":
    main()
