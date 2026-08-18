import os
import re
from typing import Dict, Any, List
from groq import Groq
from prompts.registry import get_prompt_version

def get_groq_api_key() -> str:
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
    """Direct Native Groq SDK Generator with Verified Attribution & Guardrails."""
    
    REFUSAL_PHRASES = [
        "insufficient evidence",
        "cannot find",
        "does not mention",
        "not available in the provided",
        "no information",
        "not contain information",
        "unsupported query"
    ]

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

    def verify_citations_exist(self, citations: List[str], evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Lightweight verification: Asserts cited sources exist in the retrieved evidence set."""
        valid_sources = {
            f"{c.get('metadata', {}).get('document_name', '')} -> {c.get('metadata', {}).get('section_title', '')}".strip()
            for c in evidence_chunks
        }
        
        verified = []
        unverified = []
        
        for cit in citations:
            cleaned_cit = cit.strip("[] ")
            if any(cleaned_cit in src or src in cleaned_cit for src in valid_sources):
                verified.append(cit)
            else:
                unverified.append(cit)
                
        return {
            "verified_citations": verified,
            "unverified_citations": unverified,
            "has_unverified": len(unverified) > 0
        }

    def generate_answer(self, user_query: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidence_chunks:
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "citations": [],
                "unverified_citations": [],
                "grounded": False
            }

        api_key = get_groq_api_key()
        context_block = self.format_evidence_block(evidence_chunks)
        system_prompt = self.prompt_config["system_prompt"]
        user_prompt = self.prompt_config["user_template"].format(
            context_block=context_block,
            query=user_query
        )

        if api_key:
            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0,
                    max_tokens=600
                )
                if response and response.choices:
                    answer_text = response.choices[0].message.content.strip()
                    
                    # Robust multi-phrase refusal check
                    answer_lower = answer_text.lower()
                    is_refusal = any(phrase in answer_lower for phrase in self.REFUSAL_PHRASES)

                    if is_refusal:
                        return {
                            "answer": answer_text,
                            "citations": [],
                            "unverified_citations": [],
                            "grounded": False
                        }

                    # Extract citations
                    citations = re.findall(r"\[([a-zA-Z0-9_\.\-]+(?:\s*->\s*[a-zA-Z0-9_\.\-:\s]+)?)\]", answer_text)
                    citations = list(set(citations))
                    
                    # Verify citations against evidence pool
                    cit_check = self.verify_citations_exist(citations, evidence_chunks)

                    return {
                        "answer": answer_text,
                        "citations": cit_check["verified_citations"],
                        "unverified_citations": cit_check["unverified_citations"],
                        "grounded": len(cit_check["verified_citations"]) > 0 and not cit_check["has_unverified"]
                    }
            except Exception:
                pass

        # Offline Refusal Fallback
        return {
            "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
            "citations": [],
            "unverified_citations": [],
            "grounded": False
        }