from typing import List, Dict, Any
from collections import defaultdict
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever

class HybridRetriever:
    """
    Combines Dense Vector Retrieval (Semantic) and Sparse BM25 Retrieval (Keyword)
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vector_store: VectorStoreManager, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5, dense_limit: int = 10, sparse_limit: int = 10) -> List[Dict[str, Any]]:
        """
        Executes parallel dense + sparse queries and fuses their ranked lists via RRF.
        """
        # 1. Fetch Dense Vector Results
        dense_results = self.vector_store.search(query, top_k=dense_limit)
        
        # 2. Fetch Sparse BM25 Results
        sparse_results = self.bm25_retriever.search(query, top_k=sparse_limit)

        # 3. Calculate RRF Fusion Scores
        rrf_scores = defaultdict(float)
        chunk_map = {}

        # Add Dense Ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
            chunk_map[cid] = {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "dense_rank": rank,
                "sparse_rank": None
            }

        # Add Sparse Ranks
        for rank, res in enumerate(sparse_results, start=1):
            item = res["chunk"]
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
            
            if cid in chunk_map:
                chunk_map[cid]["sparse_rank"] = rank
            else:
                chunk_map[cid] = {
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "dense_rank": None,
                    "sparse_rank": rank
                }

        # 4. Sort by Final RRF Score
        fused_results = []
        for cid, score in rrf_scores.items():
            entry = chunk_map[cid]
            entry["rrf_score"] = score
            fused_results.append(entry)

        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_results[:top_k]