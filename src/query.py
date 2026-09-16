"""Query the local Bible vector store and optionally ask the local LLM
to synthesize an answer grounded in the retrieved verses."""

import argparse

import chromadb

from . import config
from .ollama_client import chat, embed
from .rerank import rerank as rerank_hits
from .translate import to_english

SYSTEM_PROMPT = (
    "You are a careful Bible study assistant. Answer using ONLY the "
    "provided Bible passages as evidence. Always cite book, chapter and "
    "verse for every claim. If the passages don't support an answer, say so."
)

SYSTEM_PROMPT_FI = (
    "Olet huolellinen raamatuntutkimusavustaja. Vastaa VAIN annettujen "
    "raamatunkohtien perusteella ja vastaa suomeksi. Mainitse aina kirja, "
    "luku ja jae jokaiselle väitteelle. Jos kohdat eivät tue vastausta, "
    "sano niin."
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
    parser.add_argument("-k", type=int, default=None, help="Max passages to keep (default: 8, or unlimited when --min-score is used)")
    parser.add_argument("--no-answer", action="store_true", help="Only print passages, skip LLM synthesis")
    parser.add_argument("--rerank", action="store_true", help="Rerank candidates with the local LLM before keeping results")
    parser.add_argument("--fetch-k", type=int, default=None, help="Candidates to retrieve before reranking (default: 3x -k, or 60 in --min-score recall mode)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Recall mode (requires --rerank): keep every candidate scoring >= this (0-10) instead of a fixed top-k",
    )
    parser.add_argument("--lang", choices=["en", "fi"], default="en", help="Language of your query and the answer (retrieval always runs in English)")
    args = parser.parse_args()

    if args.min_score is not None and not args.rerank:
        parser.error("--min-score requires --rerank")

    search_query = args.query
    if args.lang == "fi":
        search_query = to_english(args.query)
        print(f"(haku englanniksi: {search_query})")

    recall_mode = args.min_score is not None
    default_k = 8
    fetch_k = args.fetch_k or (
        (max((args.k or default_k) * 3, 60) if recall_mode else (args.k or default_k) * 3)
        if args.rerank
        else (args.k or default_k)
    )
    hits = retrieve(search_query, fetch_k)

    if args.rerank:
        hits = rerank_hits(search_query, hits)
        if recall_mode:
            hits = [h for h in hits if h["rerank_score"] >= args.min_score]
            if args.k is not None:
                hits = hits[: args.k]
            label = f"--- Retrieved passages (reranked, score >= {args.min_score}, {len(hits)} found) ---"
        else:
            hits = hits[: args.k or default_k]
            label = "--- Retrieved passages (reranked) ---"
        print(label)
        print(format_hits(hits, show_score=True))
    else:
        hits = hits[: args.k or default_k]
        print("--- Retrieved passages ---")
        print(format_hits(hits))

    if not args.no_answer:
        print("\n--- Answer ---")
        context = format_hits(hits)
        if args.lang == "fi":
            system_prompt = SYSTEM_PROMPT_FI
            user_prompt = (
                f"Aihe/kysymys: {args.query}\n\nRaamatunkohdat (englanniksi):\n{context}\n\n"
                "Tiivistä suomeksi mitä nämä kohdat kertovat aiheesta, viitaten kohtiin."
            )
            # think=False: with a handful of passages, qwen3.5's Finnish
            # reasoning can burn through the whole token budget before
            # ever emitting an answer, leaving content empty (see
            # RERANKING.md-style note in translate.py for the same
            # failure mode). Finnish output quality is still good without it.
            answer = chat(system_prompt, user_prompt, think=False)
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = (
                f"Question/theme: {args.query}\n\nPassages:\n{context}\n\n"
                "Summarize what these passages say about the theme, citing references."
            )
            answer = chat(system_prompt, user_prompt)
        print(answer)


if __name__ == "__main__":
    main()
