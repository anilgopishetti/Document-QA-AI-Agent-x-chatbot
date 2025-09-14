"""
rag_agent.py
------------
Implements the Retrieval-Augmented Generation (RAG) pipeline:
1. Retrieve top-k relevant chunks from ChromaDB.
2. Build a context-aware prompt for Gemini.
3. Call Gemini API for an optimized answer.
4. Return structured output with sources.
"""
import argparse
import chromadb
from config import CHROMA_DB_DIR, GEMINI_API_KEY
import google.generativeai as genai
import textwrap
import tiktoken  # optional but recommended for token budgeting

# configure Gemini (already configured in config or main project)
genai.configure(api_key=GEMINI_API_KEY)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(name="documents")

def retrieve_contexts(query: str, top_k: int = 6):
    coll = get_collection()
    rr = coll.query(query_texts=[query], n_results=top_k)
    ids = rr.get("ids", [[]])[0]
    docs = rr.get("documents", [[]])[0]
    metas = rr.get("metadatas", [[]])[0]
    distances = rr.get("distances", [[]])[0]
    contexts = []
    for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
        source = meta.get("source") if isinstance(meta, dict) else meta
        contexts.append({
            "chunk_id": cid,
            "source": source,
            "rank": i+1,
            "distance": dist,
            "text": doc
        })
    return contexts

# Simple token counting helper using tiktoken (approximate)
def count_tokens(text: str, model_name="cl100k_base") -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())

def build_prompt(query: str, contexts: list, token_budget_for_context=3000):
    """
    Build a prompt that includes the user query and the top contexts (trimmed to token_budget).
    We include explicit instructions to the model to cite sources and to refuse hallucination.
    """
    intro = textwrap.dedent("""
    You are a concise research assistant. Use ONLY the provided CONTEXT snippets to answer the question.
    Cite each factual claim using the source tag in square brackets, e.g. [attention_is_all_you_need].
    If the answer is not contained in the provided context, respond: "I don't know based on the provided documents."
    Keep answers short and to the point (max 300 words).
    """).strip()

    # assemble contexts until token budget reached
    assembled = ""
    used_tokens = 0
    for c in contexts:
        c_text = f"[{c['source']}] {c['text']}\n\n"
        c_tokens = count_tokens(c_text)
        if used_tokens + c_tokens > token_budget_for_context:
            # stop or include truncated
            remaining = token_budget_for_context - used_tokens
            if remaining <= 0:
                break
            # naive truncation by characters (safe fallback)
            truncated_text = c_text[:int(len(c_text) * (remaining / c_tokens))]
            assembled += truncated_text + "\n"
            break
        assembled += c_text
        used_tokens += c_tokens

    prompt = f"{intro}\n\nCONTEXT:\n{assembled}\n\nQUESTION: {query}\n\nAnswer:"
    return prompt

def call_gemini(prompt: str, model_name="gemini-1.5-flash", max_output_tokens=512):
    """
    Call Gemini with the given prompt. Returns model text or raises on error.
    """
    model = genai.GenerativeModel(model_name)
    # generate content. API shape may vary by SDK; this uses the generate_content pattern.
    resp = model.generate_content(prompt)
    # response object may contain .text or .candidates; adjust if different SDK version
    text = getattr(resp, "text", None)
    if not text:
        # fallback: try candidates
        cand = getattr(resp, "candidates", None)
        if cand and len(cand) > 0:
            text = cand[0].get("content", "")
        else:
            text = str(resp)
    return text

def format_answer(answer_text: str, contexts: list):
    """
    Post-process answer: extract used source tags (best-effort) and present a short sources list.
    """
    # best-effort fetch all unique sources from contexts
    sources = sorted({c["source"] for c in contexts})
    return {
        "answer": answer_text.strip(),
        "sources": sources,
        "num_contexts": len(contexts)
    }

def rag_answer(query: str, top_k: int = 6):
    contexts = retrieve_contexts(query, top_k=top_k)
    if not contexts:
        return {"answer": "No relevant context found in the indexed documents.", "sources": []}
    prompt = build_prompt(query, contexts, token_budget_for_context=3000)
    # (Optional) print prompt for debugging: print(prompt[:2000]) 
    try:
        answer_text = call_gemini(prompt)
    except Exception as e:
        return {"error": f"LLM call failed: {e}", "sources": [c["source"] for c in contexts]}

    return format_answer(answer_text, contexts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=6)
    args = parser.parse_args()

    out = rag_answer(args.query, top_k=args.top_k)
    if out.get("error"):
        print("Error:", out["error"])
    else:
        print("\n=== ANSWER ===\n")
        print(out["answer"])
        print("\n=== SOURCES ===")
        for s in out["sources"]:
            print("-", s)
        print("\n(Num contexts used:", out["num_contexts"], ")")

if __name__ == "__main__":
    main()
