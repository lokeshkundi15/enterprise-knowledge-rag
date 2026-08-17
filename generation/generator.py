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
    Direct Groq SDK Grounded Answer Generator.
    Bypasses LangChain version conflicts and provides 100% resilient fallback.
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

    def _create_deterministic_fallback(self, user_query: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates a high-quality deterministic grounded answer when API is unreachable."""
        top_meta = evidence_chunks[0].get("metadata", {})
        doc = top_meta.get("document_name", "Document")
        sec = top_meta.get("section_title", "Section")
        content = evidence_chunks[0].get("text", "").strip()
        
        # Clean text
        summary = content.replace("\n", " ")[:280]
        answer = f"According to [{doc} -> {sec}], {summary}..."
        return {
            "answer": answer,
            "citations": [f"{doc} -> {sec}"],
            "grounded": True
        }

    def generate_answer(self, user_query: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidence_chunks:
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "citations": [],
                "grounded": False
            }

        api_key = get_groq_api_key()
        if not api_key:
            return self._create_deterministic_fallback(user_query, evidence_chunks)

        context_block = self.format_evidence_block(evidence_chunks)
        system_prompt = self.prompt_config["system_prompt"]
        user_prompt = self.prompt_config["user_template"].format(
            context_block=context_block,
            query=user_query
        )

        # 1. Try Primary Model: llama-3.3-70b-versatile
        # 2. Try Secondary Model: llama3-8b-8192
        candidate_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192"]

        try:
            client = Groq(api_key=api_key)
            answer_text = None

            for model in candidate_models:
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=model,
                        temperature=0.0,
                        max_tokens=600
                    )
                    answer_text = chat_completion.choices[0].message.content.strip()
                    if answer_text:
                        break
                except Exception:
                    continue

            if not answer_text:
                return self._create_deterministic_fallback(user_query, evidence_chunks)

            # Extract citations
            citations = re.findall(r"\[([a-zA-Z0-9_\.\-]+(?:\s*->\s*[a-zA-Z0-9_\.\-:\s]+)?)\]", answer_text)
            citations = list(set(citations))
            is_refusal = "insufficient evidence" in answer_text.lower()

            return {
                "answer": answer_text,
                "citations": citations,
                "grounded": not is_refusal
            }

        except Exception:
            return self._create_deterministic_fallback(user_query, evidence_chunks)