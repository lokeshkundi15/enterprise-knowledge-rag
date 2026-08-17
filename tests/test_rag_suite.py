import os
import sys
import pytest

# Add root directory to python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import DocumentReranker

@pytest.fixture(scope="session")
def setup_rag():
    chunker = DocumentChunker()
    chunks = chunker.process_directory(os.path.join(BASE_DIR, "data", "raw"))
    vstore = VectorStoreManager()
    vstore.index_chunks(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vstore, bm25)
    reranker = DocumentReranker()
    return {"chunks": chunks, "hybrid": hybrid, "reranker": reranker}

def test_document_chunker_structure(setup_rag):
    chunks = setup_rag["chunks"]
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert "text" in first_chunk
    assert "metadata" in first_chunk
    assert "document_name" in first_chunk["metadata"]
    assert "section_title" in first_chunk["metadata"]

def test_bm25_exact_code_retrieval(setup_rag):
    hybrid = setup_rag["hybrid"]
    results = hybrid.search("SEC-02 SSH Bastion Access Policy", top_k=2)
    assert len(results) > 0
    top_doc = results[0]["metadata"]["document_name"]
    top_sec = results[0]["metadata"]["section_title"]
    assert top_doc == "security_runbook.md"
    assert "SEC-02" in top_sec

def test_hybrid_rrf_scoring(setup_rag):
    hybrid = setup_rag["hybrid"]
    results = hybrid.search("What is the 401(k) retirement match?", top_k=3)
    assert len(results) > 0
    assert "rrf_score" in results[0]
    assert results[0]["rrf_score"] > 0

def test_cross_encoder_reranking_accuracy(setup_rag):
    hybrid = setup_rag["hybrid"]
    reranker = setup_rag["reranker"]
    
    query = "What is the parental leave duration for new parents?"
    candidates = hybrid.search(query, top_k=4)
    reranked = reranker.rerank(query, candidates, top_k=1)
    
    assert len(reranked) == 1
    assert reranked[0]["metadata"]["document_name"] == "hr_policy.md"
    assert "Parental Leave" in reranked[0]["metadata"]["section_title"]
    assert reranked[0]["rerank_score"] > 0.0