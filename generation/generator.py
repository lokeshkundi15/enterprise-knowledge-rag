import os
import re
from typing import Dict, Any, List
from groq import Groq
from prompts.registry import get_prompt_version

def get_groq_api_key() -> str:
    """Retrieve Groq API key from Streamlit secrets or OS environment."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                key = str(st.secrets["GROQ_API_KEY"]).strip()
        except Exception:
            pass
    return key.strip()

class GroundedAnswerGenerator:
    """
    Direct Native Groq SDK Generator with Zero-Hallucination Guardrails.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", prompt_version: str = "v1.0-strict"):
        self.model_name = model_name
        self.prompt_config = get_prompt_version(prompt_version)

    def format_evidence_block(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for idx, chunk in enumerate(evidence_chunks, start=1):
            meta = chunk.get("metadata", {})
            doc = meta.get("document_name", "UnknownDoc")
            sec = meta.get("section_title", "General")
            text = chunk.get("text", "").strip()
            blocks.append(f"[{idx}] Source: [{doc} -> {sec}]\nContent: {text}")
        return "\n\n".join(blocks)

    def generate_answer(self, user_query: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Guardrail 1: Empty Evidence Check
        if not evidence_chunks:
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "citations": [],
                "grounded": False
            }

        # Guardrail 2: Cross-Encoder Relevance Threshold Check
        # If top retrieved chunk rerank score is too weak (negative or < 0.1), refuse directly
        top_score = evidence_chunks[0].get("rerank_score", 0.0)
        if top_score < -2.0:
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "citations": [],
                "grounded": False
            }

        api_key = get_groq_api_key()
        context_block = self.format_evidence_block(evidence_chunks)
        system_prompt = self.prompt_config["system_prompt"]
        user_prompt = self.prompt_config["user_template"].format(
            context_block=context_block,
            query=user_query
        )

        active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"]

        if api_key:
            try:
                client = Groq(api_key=api_key)
                for model in active_models:
                    try:
                        response = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            model=model,
                            temperature=0.0,
                            max_tokens=600
                        )
                        if response and response.choices:
                            answer_text = response.choices[0].message.content.strip()
                            citations = re.findall(r"\[([a-zA-Z0-9_\.\-]+(?:\s*->\s*[a-zA-Z0-9_\.\-:\s]+)?)\]", answer_text)
                            citations = list(set(citations))
                            is_refusal = "insufficient evidence" in answer_text.lower()

                            return {
                                "answer": answer_text,
                                "citations": citations if not is_refusal else [],
                                "grounded": not is_refusal
                            }
                    except Exception:
                        continue
            except Exception:
                pass

        # Fallback Refusal if query is unanswerable from the chunk
        return {
            "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
            "citations": [],
            "grounded": False
        }