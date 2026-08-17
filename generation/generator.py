import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts.rag_prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE

load_dotenv()

class GroundedRAGGenerator:
    """
    Orchestrates evidence-grounded answer generation with strict citation formatting
    and hallucination prevention gates.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", min_rerank_threshold: float = -2.0):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
            
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.0,  # Zero temperature for deterministic factual extraction
            max_tokens=512
        )
        self.min_rerank_threshold = min_rerank_threshold

    def format_context(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        """Formats evidence chunks with clear metadata tags for the LLM."""
        formatted_blocks = []
        for idx, chunk in enumerate(evidence_chunks, start=1):
            doc = chunk["metadata"]["document_name"]
            sec = chunk["metadata"]["section_title"]
            cid = chunk["chunk_id"]
            
            block = f"--- [SOURCE {idx}: {doc} -> {sec} | ID: {cid}] ---\n{chunk['text']}"
            formatted_blocks.append(block)
        return "\n\n".join(formatted_blocks)

    def generate_answer(self, query: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates evidence threshold and generates a grounded response with citations.
        """
        # Safeguard Gate 1: Check if any evidence chunks exist
        if not evidence_chunks:
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "grounded": False,
                "citations": [],
                "evidence_used": []
            }

        # Safeguard Gate 2: Cross-Encoder Relevance Threshold
        best_score = evidence_chunks[0].get("rerank_score", 0.0)
        if best_score < self.min_rerank_threshold:
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "grounded": False,
                "citations": [],
                "evidence_used": [],
                "reason": f"Top evidence score ({best_score:.2f}) below threshold ({self.min_rerank_threshold})"
            }

        # Build context and prompt
        context_block = self.format_context(evidence_chunks)
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context_block=context_block,
            query=query
        )

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.invoke(messages)
        answer_text = response.content.strip()

        # Extract citations used
        citations = [
            f"{c['metadata']['document_name']} -> {c['metadata']['section_title']}"
            for c in evidence_chunks
        ]

        return {
            "answer": answer_text,
            "grounded": True,
            "citations": citations,
            "evidence_used": evidence_chunks
        }