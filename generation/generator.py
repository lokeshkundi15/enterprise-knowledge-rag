import os
import re
from typing import Dict, Any, List
from groq import Groq

def get_groq_api_key() -> str:
    """Safely fetch Groq API Key from environment or Streamlit secrets."""
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
    """Production-grade deterministic grounded answer generator with strict verification."""
    
    def __init__(self):
        self.model_name = "llama-3.3-70b-versatile"

    def generate_answer(self, user_query: str, evidence_chunks: List[Any]) -> Dict[str, Any]:
        """
        Generates grounded answer with inline citations and strict refusal guardrails.
        Never fabricates hardcoded numbers or policy rules.
        """
        refusal_msg = "I do not have sufficient evidence in the available documentation to answer this question reliably."

        # 1. Guardrail: Empty evidence check
        if not evidence_chunks or len(evidence_chunks) == 0:
            return {
                "answer": refusal_msg,
                "citations": [],
                "grounded": False,
                "refusal": True
            }

        # Extract top chunk details safely
        top_chunk = evidence_chunks[0]
        if isinstance(top_chunk, dict):
            top_score = float(top_chunk.get("rerank_score", top_chunk.get("score", 0.0)))
            doc_name = top_chunk.get("doc", top_chunk.get("doc_name", "enterprise_policy.md"))
            section_name = top_chunk.get("section", "General Policy")
        else:
            top_score = getattr(top_chunk, "rerank_score", getattr(top_chunk, "score", 0.0))
            doc_name = getattr(top_chunk, "doc", getattr(top_chunk, "doc_name", "enterprise_policy.md"))
            section_name = getattr(top_chunk, "section", "General Policy")

        # 2. Guardrail: Strict cross-encoder score threshold
        if top_score < -2.5:
            return {
                "answer": refusal_msg,
                "citations": [],
                "grounded": False,
                "refusal": True
            }

        primary_citation = f"[{doc_name} -> {section_name}]"

        # 3. LLM Inference
        api_key = get_groq_api_key()
        if not api_key:
            return {
                "answer": refusal_msg,
                "citations": [],
                "grounded": False,
                "refusal": True
            }

        context_blocks = []
        for c in evidence_chunks[:3]:
            c_doc = c.get("doc", "") if isinstance(c, dict) else getattr(c, "doc", "")
            c_sec = c.get("section", "") if isinstance(c, dict) else getattr(c, "section", "")
            c_txt = c.get("content", c.get("text", "")) if isinstance(c, dict) else getattr(c, "content", getattr(c, "text", ""))
            context_blocks.append(f"SOURCE [{c_doc} -> {c_sec}]:\n{c_txt}")

        system_prompt = (
            "You are an enterprise compliance and knowledge assistant. Answer the question strictly using ONLY the provided sources.\n"
            "Rules:\n"
            "1. You MUST cite the source inline at the end of relevant statements using format: [filename -> Section Name].\n"
            "2. If the context does not contain sufficient factual evidence to answer the query, reply with exact text: "
            "'I do not have sufficient evidence in the available documentation to answer this question reliably.'\n"
            "3. Do NOT fabricate, extrapolate, or guess policy numbers, deadlines, or rules under any circumstances."
        )

        prompt = f"CONTEXT EVIDENCE:\n{'---'.join(context_blocks)}\n\nQUESTION: {user_query}\n\nGROUNDED ANSWER:"

        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                max_tokens=400,
                timeout=10.0
            )

            if response and response.choices:
                raw_answer = response.choices[0].message.content.strip()
                
                if "insufficient evidence" in raw_answer.lower() or "do not have sufficient" in raw_answer.lower():
                    return {
                        "answer": refusal_msg,
                        "citations": [],
                        "grounded": False,
                        "refusal": True
                    }

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

        # 4. Fallback on LLM Failure: Honest refusal instead of fabricated answers
        return {
            "answer": refusal_msg,
            "citations": [],
            "grounded": False,
            "refusal": True
        }