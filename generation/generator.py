import os
import re
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from prompts.registry import get_prompt_version

class GroundedAnswerGenerator:
    """
    Production Grounded Generation Engine.
    Uses official Groq Llama-3.3-70B model with deterministic citations and zero-hallucination refusal.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", prompt_version: str = "v1.0-strict"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.prompt_config = get_prompt_version(prompt_version)
        
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.0,
            max_tokens=600
        ) if self.api_key else None

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
            return {
                "answer": "I do not have sufficient evidence in the available documentation to answer this question reliably.",
                "citations": [],
                "grounded": False
            }

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content.strip()

            # Extract inline citation references like [Doc -> Section]
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
                "answer": f"Error generating grounded answer: {str(e)}",
                "citations": [],
                "grounded": False
            }