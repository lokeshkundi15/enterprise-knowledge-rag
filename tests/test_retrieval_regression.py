import os
import sys
import json
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import DocumentReranker

DATASET_PATH = os.path.join(BASE_DIR, "evaluation", "golden_dataset.json")

@pytest.fixture(scope="session")
def setup_pipeline():
    chunker = DocumentChunker()
    chunks = chunker.process_directory(os.path.join(BASE_DIR, "data", "raw"))
    vstore = VectorStoreManager()
    vstore.index_chunks(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vstore, bm25)
    reranker = DocumentReranker()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return {"hybrid": hybrid, "reranker": reranker, "dataset": dataset}

def test_retrieval_regression_quality_gate(setup_pipeline):
    """
    CI/CD Quality Regression Gate.
    Ensures HitRate@2 >= 95% and MRR >= 0.90 across golden benchmark dataset.
    """
    hybrid = setup_pipeline["hybrid"]
    reranker = setup_pipeline["reranker"]
    dataset = setup_pipeline["dataset"]

    hits, mrr_total, count = 0, 0.0, 0

    for item in dataset:
        if not item["answerable"]:
            continue
        count += 1
        query = item["query"]
        expected_doc = item["expected_doc"]
        expected_sec = item["expected_section"]

        candidates = hybrid.search(query, top_k=6)
        reranked = reranker.rerank(query, candidates, top_k=2)

        hit_rank = 0
        for rank, res in enumerate(reranked, start=1):
            meta = res.get("metadata", {})
            if meta.get("document_name") == expected_doc and meta.get("section_title") == expected_sec:
                hit_rank = rank
                break

        if hit_rank > 0:
            hits += 1
            mrr_total += 1.0 / hit_rank

    hit_rate = (hits / count) * 100
    mrr = mrr_total / count

    # Hard CI/CD Quality Assertion Gates
    assert hit_rate >= 95.0, f"Regression Gate FAILED: HitRate@2 ({hit_rate:.1f}%) dropped below 95% threshold."
    assert mrr >= 0.90, f"Regression Gate FAILED: MRR ({mrr:.4f}) dropped below 0.90 threshold."