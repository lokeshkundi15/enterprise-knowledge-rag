import os
from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

def main():
    print("🚀 === Phase 2 & 3: Dense + Sparse Hybrid Search Verification ===")
    
    # 1. Ingest Chunks
    chunker = DocumentChunker()
    chunks = chunker.process_directory(os.path.join("data", "raw"))
    print(f"📄 Loaded {len(chunks)} document chunks.")

    # 2. Init Retrievers
    vstore = VectorStoreManager()
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector_store=vstore, bm25_retriever=bm25)

    # 3. Test Specific Challenge Queries
    test_queries = [
        ("Exact Code Match", "What is SEC-02 SSH Bastion access policy?"),
        ("Acronym / Numeric Match", "Tell me about the 401(k) company match"),
        ("Conceptual / Semantic Match", "How to take paid time off when having a new baby?")
    ]

    for label, q in test_queries:
        print(f"\n==========================================")
        print(f"🔍 [{label}] Query: '{q}'")
        print(f"==========================================")
        
        results = hybrid.search(q, top_k=2)
        for idx, res in enumerate(results, start=1):
            dense_info = f"Dense Rank: {res['dense_rank']}" if res['dense_rank'] else "Dense: Not in Top-10"
            sparse_info = f"BM25 Rank: {res['sparse_rank']}" if res['sparse_rank'] else "BM25: Not in Top-10"
            
            print(f"\n  #{idx} [RRF Score: {res['rrf_score']:.5f}] | ({dense_info}, {sparse_info})")
            print(f"     Doc: {res['metadata']['document_name']} -> {res['metadata']['section_title']}")
            print(f"     Text Preview: {res['text'][:110]}...")

if __name__ == "__main__":
    main()