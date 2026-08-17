import os
from typing import Dict, Any
from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import DocumentReranker
from generation.generator import GroundedRAGGenerator

class EnterpriseRAGPipeline:
    """End-to-End Enterprise Grounded RAG Pipeline."""
    
    def __init__(self, raw_data_dir: str = "data/raw"):
        print("⚙️ Initializing Enterprise RAG Components...")
        self.chunker = DocumentChunker()
        self.chunks = self.chunker.process_directory(raw_data_dir)
        
        self.vector_store = VectorStoreManager()
        self.vector_store.index_chunks(self.chunks)
        
        self.bm25 = BM25Retriever(self.chunks)
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25)
        self.reranker = DocumentReranker()
        self.generator = GroundedRAGGenerator()
        print("✅ Enterprise RAG Pipeline Ready.")

    def query(self, user_query: str, top_candidates: int = 5, top_evidence: int = 2) -> Dict[str, Any]:
        """Runs full query flow: Hybrid Search -> Reranking -> Safeguard Gate -> LLM Generation."""
        # 1. Hybrid Candidate Retrieval
        candidates = self.hybrid_retriever.search(user_query, top_k=top_candidates)
        
        # 2. Cross-Encoder Reranking
        evidence = self.reranker.rerank(user_query, candidates, top_k=top_evidence)
        
        # 3. Grounded Generation & Citation Formatting
        result = self.generator.generate_answer(user_query, evidence)
        return result