import os
import sys
import json
import time
from typing import List, Dict, Any
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import DocumentReranker

DATASET_PATH = os.path.join(BASE_DIR, "evaluation", "golden_dataset.json")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

class FixedChunker:
    """Fixed character length chunker (Naive baseline)."""
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        chunks = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".md", ".txt")):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        text = f.read()
                    for i in range(0, len(text), self.chunk_size - self.overlap):
                        slice_text = text[i:i + self.chunk_size]
                        if len(slice_text.strip()) > 30:
                            chunks.append({
                                "chunk_id": f"{file}#fixed_{i}",
                                "text": slice_text,
                                "metadata": {"document_name": file, "section_title": "FixedSlice", "chunk_id": f"{file}#fixed_{i}"}
                            })
        return chunks

def evaluate_retrieval(search_fn, dataset: List[Dict[str, Any]], k: int = 2) -> Dict[str, float]:
    hits, mrr_total, latencies, count = 0, 0.0, [], 0
    for item in dataset:
        if not item["answerable"]:
            continue
        count += 1
        start_t = time.time()
        results = search_fn(item["query"], top_k=k)
        latencies.append((time.time() - start_t) * 1000)

        hit_rank = 0
        for rank, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            if meta.get("document_name") == item["expected_doc"]:
                # Check section if available or text overlap
                if meta.get("section_title") == item["expected_section"] or item["expected_section"] in res["text"]:
                    hit_rank = rank
                    break
        if hit_rank > 0:
            hits += 1
            mrr_total += 1.0 / hit_rank

    return {
        f"HitRate@{k} (%)": round((hits / count) * 100, 2),
        "MRR": round(mrr_total / count, 4),
        "Avg Latency (ms)": round(sum(latencies) / len(latencies), 2)
    }

def main():
    print("🔬 === Starting Multi-Strategy Chunking Empirical Benchmark ===")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Strategy 1: Fixed Chunker
    print("⏳ Evaluating Strategy A: Fixed-Size Naive Chunking...")
    fixed_chunks = FixedChunker(chunk_size=300, overlap=50).process_directory(RAW_DATA_DIR)
    bm25_fixed = BM25Retriever(fixed_chunks)
    res_fixed = evaluate_retrieval(lambda q, top_k: [r["chunk"] for r in bm25_fixed.search(q, top_k=top_k)], dataset, k=2)
    res_fixed["Strategy"] = "A. Naive Fixed Chunking (300 chars)"
    res_fixed["Total Chunks"] = len(fixed_chunks)

    # Strategy 2: Structure-Aware Markdown Chunker
    print("⏳ Evaluating Strategy B: Structure-Aware Markdown Chunking...")
    struct_chunks = DocumentChunker().process_directory(RAW_DATA_DIR)
    vstore = VectorStoreManager()
    vstore.index_chunks(struct_chunks)
    bm25_struct = BM25Retriever(struct_chunks)
    hybrid = HybridRetriever(vstore, bm25_struct)
    reranker = DocumentReranker()
    
    res_struct = evaluate_retrieval(lambda q, top_k: reranker.rerank(q, hybrid.search(q, top_k=6), top_k=top_k), dataset, k=2)
    res_struct["Strategy"] = "B. Structure-Aware Markdown + Hybrid Reranker"
    res_struct["Total Chunks"] = len(struct_chunks)

    # Display comparison
    df = pd.DataFrame([res_fixed, res_struct])
    cols = ["Strategy", "Total Chunks", "HitRate@2 (%)", "MRR", "Avg Latency (ms)"]
    print("\n" + "="*85)
    print("📊 CHUNKING STRATEGY EMPIRICAL COMPARISON TABLE")
    print("="*85)
    print(df[cols].to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()