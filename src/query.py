"""Query the local Bible vector store and optionally ask the local LLM
to synthesize an answer grounded in the retrieved verses."""

import argparse

import chromadb

from . import config
from .ollama_client import chat, embed
from .rerank import rerank as rerank_hits

SYSTEM_PROMPT = (
    "You are a careful Bible study assistant. Answer using ONLY the "
    "provided Bible passages as evidence. Always cite book, chapter and "
    "verse for every claim. If the passages don't support an answer, say so."
)


def retrieve(query: str, k: int):
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_collection("bible")
    result = collection.query(query_embeddings=[embed(query)], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def format_hits(hits, show_score=False):
    lines = []
    for h in hits:
        m = h["meta"]
        ref = f"{m['book']} {m['chapter']}:{m['verse_start']}-{m['verse_end']}"
        prefix = f"[{ref}]"
        if show_score and "rerank_score" in h:
            prefix = f"[{ref} | score={h['rerank_score']:.1f}]"
        lines.append(f"{prefix} {h['text']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query or theme, e.g. 'God as healer'")
    parser.add_argument("-k", type=int, default=8, help="Number of passages to keep after retrieval/reranking")
    parser.add_argument("--no-answer", action="store_true", help="Only print passages, skip LLM synthesis")
    parser.add_argument("--rerank", action="store_true", help="Rerank candidates with the local LLM before keeping top-k")
    parser.add_argument("--fetch-k", type=int, default=None, help="Candidates to retrieve before reranking (default: 3x -k)")
    args = parser.parse_args()

    fetch_k = args.fetch_k or (args.k * 3 if args.rerank else args.k)
    hits = retrieve(args.query, fetch_k)

    if args.rerank:
        hits = rerank_hits(args.query, hits)[: args.k]
        print("--- Retrieved passages (reranked) ---")
        print(format_hits(hits, show_score=True))
    else:
        hits = hits[: args.k]
        print("--- Retrieved passages ---")
        print(format_hits(hits))

    if not args.no_answer:
        print("\n--- Answer ---")
        context = format_hits(hits)
        user_prompt = (
            f"Question/theme: {args.query}\n\nPassages:\n{context}\n\n"
            "Summarize what these passages say about the theme, citing references."
        )
        print(chat(SYSTEM_PROMPT, user_prompt))


if __name__ == "__main__":
    main()
