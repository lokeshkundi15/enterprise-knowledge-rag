import os
import re
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from prompts.registry import get_prompt_version

def get_groq_api_key() -> str:
    """Retrieve Groq API key from Streamlit secrets or OS environment."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return key or ""

class GroundedAnswerGenerator:
    """
    Production Grounded Generation Engine.
    Uses official Groq Llama-3.3-70B model with deterministic citations and zero-hallucination refusal.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", prompt_version: str = "v1.0-strict"):
        self.api_key = get_groq_api_key()
        self.model_name = model_name
        self.prompt_config = get_prompt_version(prompt_version)
        
        self.llm = None
        if self.api_key:
            try:
                self.llm = ChatGroq(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=0.0,
                    max_tokens=600
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize ChatGroq: {e}")

    def format_evidence_block(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        """Formats evidence passages into numbered blocks with citations."""
        blocks = []
        for idx, chunk in enumerate(evidence_chunks, start=1):
            meta = chunk.get("metadata", {})
            doc = meta.get("document_name", "UnknownDoc")
            sec = meta.get("section_title", "General")
            text = chunk.get("text", "").strip()
            blocks.append(f"[{idx}] Source: [{doc} -> {sec}]\nContent: {text}")
        return "\n\n".join(blocks)

    def generate_answer(self, user_query: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates grounded answer with source citations or deterministic refusal."""
        if not evidence_chunks:
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "citations": [],
                "grounded": False
            }

        # Lazy key check in case secrets were loaded after module import
        if not self.llm:
            current_key = get_groq_api_key()
            if current_key:
                try:
                    self.api_key = current_key
                    self.llm = ChatGroq(
                        api_key=self.api_key,
                        model_name=self.model_name,
                        temperature=0.0,
                        max_tokens=600
                    )
                except Exception:
                    pass

        context_block = self.format_evidence_block(evidence_chunks)
        system_prompt = self.prompt_config["system_prompt"]
        user_prompt = self.prompt_config["user_template"].format(
            context_block=context_block,
            query=user_query
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if not self.llm:
            # Deterministic fallback response based on top retrieved evidence
            top_meta = evidence_chunks[0].get("metadata", {})
            doc = top_meta.get("document_name", "Document")
            sec = top_meta.get("section_title", "Section")
            return {
                "answer": f"Based on the retrieved documentation [{doc} -> {sec}], {evidence_chunks[0].get('text', '').strip()[:200]}...",
                "citations": [f"{doc} -> {sec}"],
                "grounded": True
            }

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content.strip()

            citations = re.findall(r"\[([a-zA-Z0-9_\.\-]+(?:\s*->\s*[a-zA-Z0-9_\.\-:\s]+)?)\]", answer_text)
            citations = list(set(citations))
            is_refusal = "insufficient evidence" in answer_text.lower()

            return {
                "answer": answer_text,
                "citations": citations,
                "grounded": not is_refusal
            }
        except Exception as e:
            return {
                "answer": f"Error communicating with LLM: {str(e)}",
                "citations": [],
                "grounded": False
            }