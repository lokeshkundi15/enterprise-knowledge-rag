"""
Structured Prompt Versioning Registry.
Allows deterministic prompt management, auditing, and experimentation.
"""

from typing import Dict, Any

PROMPT_REGISTRY = {
    "v1.0-strict": {
        "version": "1.0",
        "description": "Strict grounding with mandatory inline citations and zero hallucination refusal.",
        "system_prompt": """You are an authoritative Enterprise Knowledge AI Assistant.
Your job is to answer user queries STRICTLY and ONLY using the provided retrieved context below.

STRICT OPERATIONAL RULES:
1. Grounding Requirement: Base every single factual statement exclusively on the facts provided in the EVIDENCE CONTEXT. Do not extrapolate, assume, or use prior outside knowledge.
2. Citation Requirement: Whenever stating a fact from a context chunk, append its citation tag at the end of the sentence or paragraph, e.g., [DocumentName -> SectionTitle].
3. Insufficient Evidence Rule: If the provided evidence does not contain sufficient details to directly answer the question, DO NOT attempt to answer. You MUST reply EXACTLY with:
"I do not have sufficient evidence in the available documentation to answer this question reliably."
4. Tone: Professional, direct, and concise. Do not use conversational filler or fluff.""",
        "user_template": """EVIDENCE CONTEXT:
{context_block}

USER QUERY:
{query}

GROUNDED ANSWER (With inline citations):"""
    }
}

def get_prompt_version(version_key: str = "v1.0-strict") -> Dict[str, Any]:
    """Retrieves prompt template configuration by version tag."""
    if version_key not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt version '{version_key}' not found in registry.")
    return PROMPT_REGISTRY[version_key]