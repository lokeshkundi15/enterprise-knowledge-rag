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
        
        # 5. Grounded Generator (Explicitly using llama-3.3-70b-versatile)
        self.generator = GroundedAnswerGenerator(model_name="llama-3.3-70b-versatile")

    def query(self, user_query: str, top_candidates: int = 5, top_evidence: int = 2) -> Dict[str, Any]:
        """Executes full RAG flow with safe fallback."""
        candidates = self.hybrid_retriever.search(user_query, top_k=top_candidates)
        evidence = self.reranker.rerank(user_query, candidates, top_k=top_evidence)
        result = self.generator.generate_answer(user_query, evidence)
        
        return {
            "query": user_query,
            "answer": result["answer"],
            "citations": result["citations"],
            "grounded": result["grounded"],
            "evidence": evidence,
            "candidates": candidates
        }