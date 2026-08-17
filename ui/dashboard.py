import os
import sys
import streamlit as st
import pandas as pd

# Add root directory to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.rag_pipeline import EnterpriseRAGPipeline

st.set_page_config(
    page_title="Enterprise Grounded RAG & Evaluation System",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource(show_spinner="Initializing Enterprise RAG Engine...")
def load_rag_pipeline():
    return EnterpriseRAGPipeline()

pipeline = load_rag_pipeline()

st.title("📚 Enterprise Grounded Knowledge Retrieval & Verification System")
st.caption("Production-Grade RAG | Hybrid BM25 + Dense Vector Search | Cross-Encoder Reranking | Zero-Hallucination Guardrails")

# Sidebar Controls
st.sidebar.header("⚙️ Retrieval & Evaluation Controls")
top_candidates = st.sidebar.slider("Hybrid Search Candidates (Top-N)", min_value=3, max_value=10, value=5)
top_evidence = st.sidebar.slider("Cross-Encoder Evidence Passages (Top-K)", min_value=1, max_value=5, value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Example Enterprise Queries")
example_queries = [
    "What is the parental leave duration for new parents?",
    "What are the exact steps to rotate production credentials?",
    "What is the maximum HikariCP pool size allowed in RFC-101?",
    "What encryption standard is mandated for customer PII and PAN?",
    "What is the company policy on bringing pets to the office? (Unanswerable)"
]

selected_query = st.sidebar.selectbox("Choose a sample query:", [""] + example_queries)

query_input = st.text_input("Ask an enterprise policy, engineering RFC, or compliance question:", value=selected_query if selected_query else "")

if st.button("🔍 Search & Generate Grounded Answer", type="primary") and query_input:
    with st.spinner("Executing Hybrid Retrieval -> Cross-Encoder Reranker -> Strict Grounding Gate -> Groq LLM..."):
        response = pipeline.query(query_input, top_candidates=top_candidates, top_evidence=top_evidence)
        
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("💡 Grounded Answer")
        if response.get("grounded"):
            st.success(response["answer"])
            
            st.markdown("#### 📚 Verified Source Citations")
            for cit in response.get("citations", []):
                st.info(f"📍 `{cit}`")
        else:
            st.error(f"🛡️ **Grounding Safeguard Triggered (Refusal to Hallucinate):**\n\n{response['answer']}")
            if "reason" in response:
                st.caption(f"Reason: {response['reason']}")

    with col2:
        st.subheader("🔬 Evidence & Reranker Inspection")
        evidence = response.get("evidence_used", [])
        if evidence:
            evidence_data = []
            for e in evidence:
                evidence_data.append({
                    "Doc": e["metadata"]["document_name"],
                    "Section": e["metadata"]["section_title"],
                    "Rerank Score": f"{e.get('rerank_score', 0.0):.4f}",
                    "RRF Score": f"{e.get('rrf_score', 0.0):.4f}"
                })
            st.dataframe(pd.DataFrame(evidence_data), use_container_width=True)
            
            with st.expander("📄 View Extracted Raw Chunks"):
                for idx, e in enumerate(evidence, 1):
                    st.markdown(f"**Chunk #{idx} [{e['metadata']['document_name']} -> {e['metadata']['section_title']}]:**")
                    st.code(e["text"], language="markdown")
        else:
            st.warning("No candidate evidence chunks met the relevance threshold.")