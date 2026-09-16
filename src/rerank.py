"""LLM-based reranking: ask the local chat model to score how relevant
each retrieved passage is to the query, then reorder by that score.

This is a cross-encoder substitute that stays inside the Ollama stack
(no extra torch/sentence-transformers dependency) at the cost of being
slower per-candidate than a real cross-encoder model.
"""

import re

from .ollama_client import chat

RERANK_SYSTEM = (
    "You score how relevant a Bible passage is to a search theme. "
    "Respond with ONLY an integer from 0 to 10, nothing else. "
    "10 = the passage directly and clearly speaks to the theme, even if "
    "it uses different words. 0 = unrelated."
)


def score_passage(query: str, passage_text: str) -> float:
    prompt = f"Theme: {query}\n\nPassage: {passage_text}\n\nRelevance score (0-10):"
    reply = chat(RERANK_SYSTEM, prompt, think=False)
    match = re.search(r"\d+(\.\d+)?", reply)
    if not match:
        return 0.0
    return min(float(match.group()), 10.0)


def rerank(query: str, hits: list[dict]) -> list[dict]:
    scored = []
    for h in hits:
        score = score_passage(query, h["text"])
        scored.append({**h, "rerank_score": score})
    scored.sort(key=lambda h: h["rerank_score"], reverse=True)
    return scored
