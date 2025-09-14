# src/arxiv_helper.py
import arxiv
import re

def search_arxiv(query, max_results=3):
    """
    Search Arxiv for papers matching the query.
    Returns a list of dicts with title, authors, summary, url.
    """
    # Clean query: remove phrases like 'find paper', 'arxiv', etc.
    cleaned_query = re.sub(r'\b(find|paper|arxiv|on|about)\b', '', query, flags=re.IGNORECASE).strip()

    if not cleaned_query:
        cleaned_query = query  # fallback to original query if cleaning removed everything

    search = arxiv.Search(
        query=cleaned_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for paper in search.results():
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "url": paper.entry_id
            })
    except arxiv.UnexpectedEmptyPageError:
        # Handle empty pages gracefully
        pass

    return results
