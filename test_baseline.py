import os
from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager

def main():
    print("🚀 === Phase 1: Baseline RAG Verification ===")
    
    # 1. Chunk documents
    chunker = DocumentChunker()
    raw_data_dir = os.path.join("data", "raw")
    chunks = chunker.process_directory(raw_data_dir)
    print(f"📄 Loaded & chunked {len(chunks)} document sections.")
    
    # 2. Index into ChromaDB
    vstore = VectorStoreManager()
    vstore.index_chunks(chunks)
    
    # 3. Test Retrieval
    queries = [
        "What is the parental leave policy?",
        "How do we rotate production secrets in AWS?",
        "What is the home office stipend limit?"
    ]
    
    print("\n🔍 Testing Semantic Vector Search Queries:")
    for q in queries:
        print(f"\n❓ Query: {q}")
        results = vstore.search(q, top_k=1)
        if results:
            match = results[0]
            print(f"   🎯 Match Found: [{match['metadata']['document_name']} -> {match['metadata']['section_title']}]")
            print(f"   📝 Evidence Preview: {match['text'][:120]}...")
            print(f"   📏 Cosine Distance: {match.get('distance', 'N/A'):.4f}")

if __name__ == "__main__":
    main()