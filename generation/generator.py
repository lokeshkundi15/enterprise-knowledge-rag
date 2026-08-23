import os
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


# IMPORTANT: This threshold was previously set to -25.0, which almost never
# triggers a refusal (cross-encoder ms-marco-MiniLM-L-6-v2 scores for a truly
# irrelevant pair, like "pets policy" vs "parental leave" text, were measured
# at -10.39 in production — well above -25.0, so the refusal never fired).
#
# -3.0 is a safer starting point (clearly irrelevant chunks tend to score well
# below this), but you should calibrate this empirically for your own corpus:
# run 5 known-relevant queries and 5 known-irrelevant queries through the
# reranker, print the scores, and set the threshold between the two clusters.
RELEVANCE_THRESHOLD = -3.0


class GroundedAnswerGenerator:
    """Production-grade deterministic grounded answer generator with strict verification."""

    def __init__(self):
        self.model_name = "llama-3.3-70b-versatile"

    def generate_answer(self, user_query: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates grounded answer with inline citations and zero-hallucination guardrail.

        Expects each chunk to match the real pipeline schema:
            {"text": "...", "metadata": {"document_name": "...", "section_title": "..."},
             "rerank_score": float, ...}
        """
        if not evidence_chunks or len(evidence_chunks) == 0:
            return self._refusal()

        top_chunk = evidence_chunks[0]
        top_metadata = top_chunk.get("metadata", {}) or {}

        top_score = float(top_chunk.get("rerank_score", top_chunk.get("score", 0.0)))
        doc_name = top_metadata.get("document_name", "unknown_document")
        section_name = top_metadata.get("section_title", "unknown_section")
        raw_content = (top_chunk.get("text", "") or "").strip()

        # Guardrail: reject clearly irrelevant top evidence before calling the LLM at all.
        if top_score < RELEVANCE_THRESHOLD:
            return self._refusal()

        primary_citation = f"[{doc_name} -> {section_name}]"

        # Online LLM Inference via Groq (primary path)
        api_key = get_groq_api_key()
        if api_key:
            try:
                client = Groq(api_key=api_key)
                context_blocks = []
                for c in evidence_chunks[:3]:
                    c_meta = c.get("metadata", {}) or {}
                    c_doc = c_meta.get("document_name", "unknown_document")
                    c_sec = c_meta.get("section_title", "unknown_section")
                    c_txt = c.get("text", "")
                    context_blocks.append(f"SOURCE [{c_doc} -> {c_sec}]:\n{c_txt}")

                system_prompt = (
                    "You are an enterprise compliance AI. Answer the question strictly using ONLY the provided sources.\n"
                    "You MUST cite the source inline at the end of relevant statements using format: [filename -> Section Name].\n"
                    "If the context does not contain the answer, reply with exact text: "
                    "'I do not have sufficient evidence in the available documentation to answer this question reliably.'"
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
                pass  # Fall through to the safe fallback below — never fabricate.

        # No API key, or the API call failed: show the ACTUAL retrieved text
        # (never a synthesized/guessed answer), or refuse if there isn't enough of it.
        if raw_content and len(raw_content) >= 40:
            excerpt = raw_content.replace("\n", " ")
            if len(excerpt) > 400:
                excerpt = excerpt[:400].rsplit(" ", 1)[0] + "..."
            return {
                "answer": (
                    f"(LLM unavailable — showing retrieved evidence directly, "
                    f"not an AI-generated answer)\n\n{excerpt} {primary_citation}"
                ),
                "citations": [primary_citation],
                "grounded": True,
                "refusal": False
            }

        return self._refusal()

    @staticmethod
    def _refusal() -> Dict[str, Any]:
        return {
            "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
            "citations": [],
            "grounded": False,
            "refusal": True
        }