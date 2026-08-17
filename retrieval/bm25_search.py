import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """Sparse lexical search engine using BM25Okapi for exact keyword matching."""
    
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes and standardizes text into lowercase alphanumeric tokens."""
        text = text.lower()
        # Tokenize by words, keeping hyphenated codes and alphanumeric terms
        tokens = re.findall(r'[a-zA-Z0-9\-\(\)]+', text)
        return tokens

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns top-k chunks matching the query using BM25 scoring."""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Pair chunk with its score and sort
        ranked_results = []
        for idx, score in enumerate(scores):
            if score > 0:  # Only consider chunks with non-zero keyword match
                ranked_results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score)
                })
                
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results[:top_k]