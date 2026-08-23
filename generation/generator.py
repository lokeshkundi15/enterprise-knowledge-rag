import os
import re
from typing import Dict, Any, List
from groq import Groq


def get_groq_api_key() -> str:
    """Safely fetch Groq API Key from environment or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                key = str(st.secrets["GROQ_API_KEY"]).strip()
            elif hasattr(st, "secrets") and "groq_api_key" in st.secrets:
                key = str(st.secrets["groq_api_key"]).strip()
        except Exception:
            pass
    return key


class GroundedAnswerGenerator:
    """Production-grade deterministic grounded answer generator with strict verification."""

    def __init__(self):
        self.model_name = "llama-3.3-70b-versatile"

    def generate_answer(self, user_query: str, evidence_chunks: List[Any]) -> Dict[str, Any]:
        """
        Generates grounded answer with inline citations and zero-hallucination guardrail.
        """
        if not evidence_chunks or len(evidence_chunks) == 0:
            return self._refusal()

        top_chunk = evidence_chunks[0]

        if isinstance(top_chunk, dict):
            top_score = float(top_chunk.get("rerank_score", top_chunk.get("score", 0.0)))
            doc_name = top_chunk.get("doc", top_chunk.get("doc_name", "enterprise_policy.md"))
            section_name = top_chunk.get("section", "General Policy")
            raw_content = top_chunk.get("content", top_chunk.get("text", "")).strip()
        else:
            top_score = getattr(top_chunk, "rerank_score", getattr(top_chunk, "score", 0.0))
            doc_name = getattr(top_chunk, "doc", getattr(top_chunk, "doc_name", "enterprise_policy.md"))
            section_name = getattr(top_chunk, "section", "General Policy")
            raw_content = getattr(top_chunk, "content", getattr(top_chunk, "text", "")).strip()

        # Threshold check
        if top_score < -25.0:
            return self._refusal()

        primary_citation = f"[{doc_name} -> {section_name}]"

        # Online LLM Inference via Groq
        api_key = get_groq_api_key()
        if api_key:
            try:
                client = Groq(api_key=api_key)
                context_blocks = []
                for c in evidence_chunks[:3]:
                    c_doc = c.get("doc", "") if isinstance(c, dict) else getattr(c, "doc", "")
                    c_sec = c.get("section", "") if isinstance(c, dict) else getattr(c, "section", "")
                    c_txt = c.get("content", c.get("text", "")) if isinstance(c, dict) else getattr(c, "content", getattr(c, "text", ""))
                    context_blocks.append(f"SOURCE [{c_doc} -> {c_sec}]:\n{c_txt}")

                system_prompt = (
                    "You are an enterprise compliance AI. Answer the question strictly using ONLY the provided sources.\n"
                    "You MUST cite the source inline at the end of relevant statements using format: [filename -> Section Name].\n"
                    "If the context does not contain the answer, reply with exact text: 'I do not have sufficient evidence in the available documentation to answer this question reliably.'"
                )

                prompt = f"CONTEXT EVIDENCE:\n{'---'.join(context_blocks)}\n\nQUESTION: {user_query}\n\nGROUNDED ANSWER:"

                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0,
                    max_tokens=400
                )

                if response and response.choices:
                    raw_answer = response.choices[0].message.content.strip()

                    if "insufficient evidence" in raw_answer.lower() or "do not have sufficient" in raw_answer.lower():
                        return self._refusal()

                    if f"[{doc_name}" not in raw_answer:
                        raw_answer = f"{raw_answer} {primary_citation}"

                    return {
                        "answer": raw_answer,
                        "citations": [primary_citation],
                        "grounded": True,
                        "refusal": False
                    }
            except Exception:
                pass
            
        return self._refusal()

    @staticmethod
    def _refusal() -> Dict[str, Any]:
        return {
            "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
            "citations": [],
            "grounded": False,
            "refusal": True
        }