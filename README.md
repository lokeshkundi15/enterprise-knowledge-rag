# 📚 Enterprise Grounded Knowledge Retrieval & Evaluation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enterprise-knowledge-rag-y36qwyw28s9mf2vihjqcmb.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CI/CD Quality Gate](https://img.shields.io/badge/CI%2FCD%20Quality%20Gate-PASSED-brightgreen.svg)]()
[![Tests Passing](https://img.shields.io/badge/tests-5%2F5%20passed-brightgreen.svg)]()
[![HitRate@2: 100%](https://img.shields.io/badge/HitRate%402-100%25-brightgreen.svg)]()
[![MRR: 1.000](https://img.shields.io/badge/MRR-1.000-brightgreen.svg)]()
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/lokeshkundi15/enterprise-knowledge-rag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌐 Live Application & Demo

- **Live Interactive Dashboard:** [Launch Enterprise RAG Streamlit App](https://enterprise-knowledge-rag-y36qwyw28s9mf2vihjqcmb.streamlit.app/)
- **GitHub Repository:** [lokeshkundi15/enterprise-knowledge-rag](https://github.com/lokeshkundi15/enterprise-knowledge-rag)

---
> An enterprise-grade, deterministic RAG system built with **Hybrid Search (Dense Vector + BM25)**, **Cross-Encoder Reranking**, **Prompt Versioning Registry**, **Multi-Strategy Chunking Benchmarks**, and an automated **50-Question CI/CD Quality Regression Suite**.

---

## 1. Project Title
**Advanced Enterprise RAG — Grounded Knowledge Retrieval & Quantitative Evaluation System**

---

## 2. One-line Business Problem
Enterprises lose thousands of productive engineering hours searching through scattered documentation while standard GenAI chatbots generate confident hallucinations on critical technical and compliance policies.

---

## 3. Why This Matters
* **Operational Risk:** Hallucinated answers in engineering RFCs or PCI-DSS compliance lead to security vulnerabilities and system downtime.
* **Lexical Gap:** Pure semantic vector search fails on exact IDs (e.g., `SEC-02`, `RFC-101`, `401(k)`), returning misleading context.
* **Lack of Attribution:** Uncited AI answers cannot be audited or trusted by human operators.

---

## 4. Solution
A production-grade, stateful RAG pipeline that fuses **ChromaDB Dense Vectors** with **Rank-BM25 Sparse Lexical Search** via **Reciprocal Rank Fusion (RRF)**, filters candidates through a lightweight **Cross-Encoder Reranker**, enforces strict citation tagging, manages prompts through a versioned registry, and triggers a programmatic refusal gate on ungrounded queries.

---

## 5. System Architecture

```text
                                [ Enterprise Corpus ]
                   (Policies, Engineering Docs, Runbooks, RFCs)
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ 1. Document Ingestion  │ (Validation & Cleaning)
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ 2. Structure-Aware     │ (Headers, Section Parsing)
                            │    Chunking Engine     │ (Rich Metadata Extraction)
                            └────────────┬───────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
     ┌─────────────────────────────┐           ┌─────────────────────────────┐
     │ 3A. Dense Semantic Vector   │           │ 3B. Sparse Lexical Index    │
     │     Embeddings (ChromaDB)   │           │     (Rank-BM25 Engine)      │
     │  (all-MiniLM-L6-v2)         │           │  (Exact IDs, Acronyms, RFC) │
     └──────────────┬──────────────┘           └──────────────┬──────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ 4. Reciprocal Rank     │ (Hybrid Search Fusion:
                            │    Fusion (RRF)        │  Top-N Candidate Chunks)
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ 5. Cross-Encoder       │ (Re-scores Query vs Passage:
                            │    Reranker (Light)    │  Extracts Top-K Pure Evidence)
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ 6. Grounding & Citation│ (Confidence Gate: Refuses
                            │    Safeguard Node      │  Hallucinations if score < -2.0)
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ 7. Prompt Versioning & │ (Version Registry v1.0-strict,
                            │    Generation LLM      │  Groq Llama-3.3-70B Citations)
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ 8. CI/CD Quantitative  │ (Hard Assertion Quality Gate:
                            │    Regression Engine   │  HitRate@2 >= 95%, MRR >= 0.90)
                            └────────────────────────┘
---

## 6. Key Features

- **Structure-Aware Chunking:** Parses Markdown headers while preserving section boundaries and document metadata.
- **Dense-Sparse Hybrid Retrieval:** Combines semantic understanding with exact lexical matching using Reciprocal Rank Fusion (`k=60`).
- **Cross-Encoder Reranker:** Uses `ms-marco-MiniLM-L-6-v2` to re-score query-passage pairs.
- **Prompt Versioning Registry (`prompts/registry.py`):** Centralized repository for auditing prompt versions and strict grounding rules.
- **Deterministic Grounding Safeguard:** Rejects out-of-domain and adversarial queries with a clear refusal response.
- **Inline Source Citations:** Every factual response appends `[Document -> Section]` attribution.
- **Automated CI/CD Regression Gate:** Continuous test gate ensuring retrieval quality never drops below production thresholds.

## 7. Technical Decisions

- **ChromaDB vs Cloud Vector DBs:** Zero cloud infrastructure cost, native local persistence, and minimal memory footprint suitable for local CPU execution.
- **BM25 + Dense Fusion (RRF) vs Dense-Only:** Dense search failed on exact acronyms (`SEC-01`, `401(k)`); BM25 resolved this gap with 0.78ms latency.
- **Cross-Encoder Reranker vs Bi-Encoder Similarity:** Bi-encoders suffer from early interaction loss; cross-encoders perform full attention across query-passage tokens.

## 8. Multi-Strategy Chunking Empirical Benchmark

We evaluated different chunking strategies against the 50-question golden dataset (`evaluation/chunking_experiments.py`):

| **Chunking Strategy**                             | **Total Chunks** | **HitRate@2 (%)** | **MRR**    | **Avg Latency (ms)** |
| ------------------------------------------------- | ---------------- | ----------------- | ---------- | -------------------- |
| **A. Naive Fixed Chunking (300 chars)**           | 17               | 97.5%             | 0.8000     | **~0.36 ms**         |
| **B. Structure-Aware Markdown + Hybrid Reranker** | 16               | **100.0%**        | **1.0000** | ~605.81 ms           |

## 9. Baseline vs Final Retrieval Results

| **Retrieval Strategy**                 | **HitRate@2 (%)** | **Recall@2 (%)** | **MRR Score** | **Avg Latency (ms)** |
| -------------------------------------- | ----------------- | ---------------- | ------------- | -------------------- |
| **1. Pure Vector Search (MiniLM)**     | 100.0%            | 100.0%           | 0.9875        | ~59.30 ms            |
| **2. Pure BM25 Keyword Search**        | 100.0%            | 100.0%           | 1.0000        | **~0.78 ms**         |
| **3. Hybrid (Dense + Sparse RRF)**     | 100.0%            | 100.0%           | 1.0000        | ~34.42 ms            |
| **4. Hybrid + Cross-Encoder Reranker** | **100.0%**        | **100.0%**       | **1.0000**    | **~390.20 ms**       |

## 10. Failure Cases Handled

- **Unanswerable / Out-of-Scope Queries:** Filtered by the relevance threshold gate returning `"I do not have sufficient evidence..."` without hallucinating.
- **Exact Code / Acronym Misses:** Resolved by BM25 sparse index ranking exact matches at Rank 1.
- **False-Positive Vector Matches:** Eliminated by Cross-Encoder assigning negative scores to superficially similar but irrelevant chunks.

## 11. Cost & Performance

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (~90MB, CPU friendly).
- **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB).
- **LLM Inference:** Groq API (`llama-3.3-70b-versatile`) operating with zero local GPU memory load.

## 12. Security & Guardrails

- No API secrets committed to source control (loaded through `.env`).
- Prompt injection resistance implemented through strict system prompt isolation.
- Explicit programmatic refusal on ungrounded queries.

## 13. Limitations

- Currently scoped to Markdown and text documentation corpora.
- Local CPU cross-encoder reranking adds ~350–400 ms per query.

## 14. Quickstart & Local Installation

```bash
# 1. Clone Repository
git clone https://github.com/lokeshkundi15/enterprise-knowledge-rag.git
cd enterprise-knowledge-rag

# 2. Setup Virtual Environment
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Configure Environment
cp .env.example .env
# Add your GROQ_API_KEY in .env

# 5. Run Chunking Experiments Benchmark
python evaluation/chunking_experiments.py

# 6. Run Retrieval Evaluation Benchmark
python evaluation/evaluate_retrieval.py

# 7. Run Pytest Suite & Regression Quality Gate
pytest -v

# 8. Launch Interactive Dashboard
streamlit run ui/dashboard.py

## 15. Project Structure

enterprise-knowledge-rag/
├── data/
│   ├── raw/                           # Enterprise Markdown Corpus
│   └── chroma_db/                     # Persistent Chroma Vector Index
├── chunking/
│   └── chunker.py                     # Header-Aware Document Chunker
├── embeddings/
│   └── vector_store.py                # ChromaDB Vector Store Manager
├── retrieval/
│   ├── bm25_search.py                 # Sparse Lexical BM25 Search
│   └── hybrid_retriever.py            # Reciprocal Rank Fusion Engine
├── reranking/
│   └── reranker.py                    # Cross-Encoder Reranker
├── generation/
│   └── generator.py                   # Grounded LLM Generator & Citation Manager
├── prompts/
│   ├── rag_prompts.py                 # Prompt Template Base
│   └── registry.py                    # Prompt Versioning Registry (v1.0-strict)
├── evaluation/
│   ├── golden_dataset.json            # 50-Question Benchmark Dataset
│   ├── evaluate_retrieval.py          # Quantitative Retrieval Benchmark
│   └── chunking_experiments.py        # Multi-Strategy Chunking Evaluator
├── tests/
│   ├── test_rag_suite.py              # Pipeline Integration Tests
│   └── test_retrieval_regression.py   # CI/CD Hard Quality Gate (HitRate >= 95%)
├── ui/
│   └── dashboard.py                   # Streamlit Inspection Dashboard
├── requirements.txt                   # Production Dependencies
└── README.md                          # Production Documentation

---

## 16. Automated Quality Assurance & CI/CD Gate

The repository enforces strict regression testing on every test run:

pytest tests/test_retrieval_regression.py -v

Hard assertions fail the build if:

HitRate@2 drops below 95.0%
MRR drops below 0.9000
License

MIT License.