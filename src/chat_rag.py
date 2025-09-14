"""
chat_rag.py
-----------
Interactive CLI chatbot that:
- Retrieves answers from local RAG pipeline.
- Integrates Arxiv search for external papers.
"""
import argparse
from rag_agent import rag_answer
from arxiv_helper import search_arxiv
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

def chat(top_k: int = 5):
    print(Fore.CYAN + "\n RAG Chatbot ready! Ask questions about your documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("Tip: Use 'find paper' or include 'arxiv' in your query to search Arxiv.\n")

    while True:
        query = input(Fore.GREEN + "You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print(Fore.MAGENTA + " Goodbye!")
            break
        if not query:
            continue

        # --- Arxiv search integration ---
        if query.lower().startswith("find paper") or "arxiv" in query.lower():
            papers = search_arxiv(query)
            if papers:
                print(Fore.CYAN + "\n Arxiv search results:")
                for i, p in enumerate(papers, 1):
                    print(Fore.YELLOW + f"{i}. {p['title']} by {', '.join(p['authors'])}")
                    print(Fore.BLUE + f"URL: {p['url']}")
                    print(Fore.WHITE + f"Summary: {p['summary'][:300]}...\n")
            else:
                print(Fore.RED + "No papers found on Arxiv for this query.")
            continue

        # --- Normal RAG retrieval ---
        response = rag_answer(query, top_k=top_k)

        if response.get("error"):
            print(Fore.RED + " Error:", response["error"])
        else:
            print(Fore.YELLOW + "\n Answer:\n" + Fore.WHITE + response["answer"])
            print(Fore.BLUE + "\n Sources: " + Fore.WHITE + ", ".join(response["sources"]))
            print(Fore.MAGENTA + "-----")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5, help="Number of document chunks to retrieve")
    args = parser.parse_args()

    chat(top_k=args.top_k)
