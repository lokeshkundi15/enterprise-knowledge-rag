import os
from typing import Dict, Any, List
from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import DocumentReranker
from generation.generator import GroundedAnswerGenerator

class EnterpriseRAGPipeline:
    """End-to-End Enterprise RAG Pipeline using Hybrid Search, Reranking, and Grounded Generation."""
    def __init__(self, raw_docs_dir: str = "data/raw"):
        self.chunker = DocumentChunker()
        self.chunks = self.chunker.process_directory(raw_docs_dir)
        
        # 1. Dense Vector Store
        self.vector_store = VectorStoreManager()
        self.vector_store.index_chunks(self.chunks)
        
        # 2. Sparse BM25 Search
        self.bm25_retriever = BM25Retriever(self.chunks)
        
        # 3. Hybrid Fusion
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_retriever)
        
        # 4. Cross-Encoder Reranker
        self.reranker = DocumentReranker()
        
        # 5. Grounded Generator
        self.generator = GroundedAnswerGenerator()

    def query(self, user_query: str, top_candidates: int = 5, top_evidence: int = 2) -> Dict[str, Any]:
        """Executes full RAG flow with strict Cross-Encoder Relevance Threshold."""
        candidates = self.hybrid_retriever.search(user_query, top_k=top_candidates)
        evidence = self.reranker.rerank(user_query, candidates, top_k=top_evidence)
        
        # Strict Relevance Threshold Gate:
        # If top rerank score is strongly negative (< -2.0), reject to prevent hallucinations
        if evidence and evidence[0].get("rerank_score", 0.0) < -2.0:
            return {
                "query": user_query,
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "citations": [],
                "grounded": False,
                "reason": "Top retrieved chunk scored below semantic relevance threshold (Cross-Encoder score < -2.0).",
                "evidence": [],
                "candidates": candidates
            }

        result = self.generator.generate_answer(user_query, evidence)
        
        return {
            "query": user_query,
            "answer": result["answer"],
            "citations": result["citations"],
            "grounded": result["grounded"],
            "evidence": evidence,
            "candidates": candidates
        }