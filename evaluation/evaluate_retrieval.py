import os
import sys
import json
import time
from typing import List, Dict, Any
import pandas as pd

# Add root directory to python path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from chunking.chunker import DocumentChunker
from embeddings.vector_store import VectorStoreManager
from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import DocumentReranker

DATASET_PATH = os.path.join(BASE_DIR, "evaluation", "golden_dataset.json")

def evaluate_strategy(name: str, search_fn, dataset: List[Dict[str, Any]], k: int = 3) -> Dict[str, float]:
    """Evaluates HitRate@K, Recall@K, and MRR for answerable questions in dataset."""
    hits = 0
    mrr_total = 0.0
    latencies = []
    answerable_count = 0

    for item in dataset:
        if not item["answerable"]:
            continue  # Evaluate retrieval only on queries where evidence exists
        
        answerable_count += 1
        query = item["query"]
        expected_doc = item["expected_doc"]
        expected_sec = item["expected_section"]

        start_time = time.time()
        results = search_fn(query, top_k=k)
        latencies.append((time.time() - start_time) * 1000)

        # Check if expected document + section is retrieved
        hit_rank = 0
        for rank, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            if meta.get("document_name") == expected_doc and meta.get("section_title") == expected_sec:
                hit_rank = rank
                break

        if hit_rank > 0:
            hits += 1
            mrr_total += 1.0 / hit_rank

    hit_rate = (hits / answerable_count) * 100 if answerable_count > 0 else 0.0
    mrr = (mrr_total / answerable_count) if answerable_count > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "Strategy": name,
        f"HitRate@{k} (%)": round(hit_rate, 2),
        f"Recall@{k} (%)": round(hit_rate, 2),
        "MRR Score": round(mrr, 4),
        "Avg Latency (ms)": round(avg_latency, 2)
    }

def main():
    print("🔬 === Phase 6 & 7: Quantitative Retrieval Evaluation Benchmark ===")
    
    raw_data_dir = os.path.join(BASE_DIR, "data", "raw")
    
    # 1. Load Chunks
    chunker = DocumentChunker()
    chunks = chunker.process_directory(raw_data_dir)
    print(f"📄 Total Indexed Chunks: {len(chunks)}")

    # 2. Initialize Engines
    vstore = VectorStoreManager()
    vstore.index_chunks(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vstore, bm25)
    reranker = DocumentReranker()

    # 3. Load Golden Dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"🧪 Loaded Golden Evaluation Dataset: {len(dataset)} questions ({len([d for d in dataset if d['answerable']])} answerable).")

    # Define Retrieval Strategies
    strategies = [
        ("1. Pure Vector Search (MiniLM)", lambda q, top_k: vstore.search(q, top_k=top_k)),
        ("2. Pure BM25 Keyword Search", lambda q, top_k: [r["chunk"] for r in bm25.search(q, top_k=top_k)]),
        ("3. Hybrid (Dense + Sparse RRF)", lambda q, top_k: hybrid.search(q, top_k=top_k)),
        ("4. Hybrid + Cross-Encoder Reranker", lambda q, top_k: reranker.rerank(q, hybrid.search(q, top_k=6), top_k=top_k))
    ]

    benchmark_rows = []
    for name, fn in strategies:
        print(f"   ⏳ Benchmarking: {name}...")
        metrics = evaluate_strategy(name, fn, dataset, k=2)
        benchmark_rows.append(metrics)

    # 4. Display Formatted Benchmark Comparison Table
    df = pd.DataFrame(benchmark_rows)
    print("\n" + "="*85)
    print("🏆 EMPIRICAL RETRIEVAL BENCHMARK RESULTS (50-QUESTION GOLDEN DATASET)")
    print("="*85)
    print(df.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    main()