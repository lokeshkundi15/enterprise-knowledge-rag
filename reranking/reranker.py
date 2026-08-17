from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class DocumentReranker:
    """
    Lightweight Cross-Encoder Reranker that re-evaluates and scores 
    Candidate Chunks against the user query for maximum precision.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        # Loads lightweight ~80MB model optimized for CPU
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes hybrid retrieval candidates, predicts cross-attention relevance scores,
        and returns the top-k most relevant evidence chunks.
        """
        if not candidate_chunks:
            return []

        # Prepare (query, text) pairs for the cross-encoder
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
        
        # Predict relevance scores (higher = more relevant)
        scores = self.model.predict(pairs)

        # Attach rerank score to chunks
        reranked = []
        for idx, chunk in enumerate(candidate_chunks):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(scores[idx])
            reranked.append(chunk_copy)

        # Sort descending by cross-encoder score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]