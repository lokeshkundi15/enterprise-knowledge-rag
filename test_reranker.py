import os
from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import DocumentReranker

def main():
    print("🚀 === Phase 4: Cross-Encoder Reranker Verification ===")
    
    # 1. Setup Retrieval Pipeline
    chunker = DocumentChunker()
    chunks = chunker.process_directory(os.path.join("data", "raw"))
    
    vstore = VectorStoreManager()
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector_store=vstore, bm25_retriever=bm25)
    reranker = DocumentReranker()

    query = "What happens to employee health insurance during parental leave?"
    print(f"\n🔍 Query: '{query}'")

    # Step A: Hybrid Search retrieves Top-5 Candidates
    print("\n--- Step 1: Hybrid Retrieval Candidates (Top 4) ---")
    candidates = hybrid.search(query, top_k=4)
    for idx, c in enumerate(candidates, start=1):
        print(f"  Candidate #{idx} [RRF Score: {c['rrf_score']:.4f}] -> {c['metadata']['section_title']}")

    # Step B: Cross-Encoder Reranks to Top-2 Evidence
    print("\n--- Step 2: Cross-Encoder Reranked Evidence (Top 2) ---")
    reranked = reranker.rerank(query, candidates, top_k=2)
    for idx, r in enumerate(reranked, start=1):
        print(f"\n  🎯 Rank #{idx} [Rerank Score: {r['rerank_score']:.4f}]")
        print(f"     Document: {r['metadata']['document_name']} | Section: {r['metadata']['section_title']}")
        print(f"     Evidence Snippet: {r['text'][:140]}...")

if __name__ == "__main__":
    main()