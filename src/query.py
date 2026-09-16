"""Query the local Bible vector store and optionally ask the local LLM
to synthesize an answer grounded in the retrieved verses."""

import argparse

import chromadb

from . import config
from .books import BOOK_NAMES_BY_LANG
from .lang_lookup import localized_text
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

OT_MAX_BOOK_NUM = 39  # 1=Genesis..39=Malachi are Old Testament, 40=Matthew..66=Revelation are New


def retrieve(query: str, k: int, where: dict | None = None):
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_collection("bible")
    result = collection.query(query_embeddings=[embed(query)], n_results=k, where=where)
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def retrieve_both_testaments(query: str, k_per_testament: int):
    """Query the Old and New Testament halves of the collection
    separately (via a book_num metadata filter) instead of one pooled
    query. A single pooled query can end up dominated by one testament
    when its vocabulary happens to sit closer to the query in embedding
    space - by the time you'd notice and filter afterwards, the other
    testament's candidates were never fetched at all."""
    old = retrieve(query, k_per_testament, where={"book_num": {"$lte": OT_MAX_BOOK_NUM}})
    new = retrieve(query, k_per_testament, where={"book_num": {"$gte": OT_MAX_BOOK_NUM + 1}})
    return old, new


def split_by_testament(hits):
    old = [h for h in hits if h["meta"]["book_num"] <= OT_MAX_BOOK_NUM]
    new = [h for h in hits if h["meta"]["book_num"] > OT_MAX_BOOK_NUM]
    return old, new


def balance_take(old_hits, new_hits, k):
    """Take up to k hits split as evenly as possible between the two
    (each already ordered by relevance), spilling over into whichever
    side has more candidates if the other comes up short."""
    n_old = (k + 1) // 2
    n_new = k - n_old
    picked_old = old_hits[:n_old]
    picked_new = new_hits[:n_new]
    deficit = (n_old - len(picked_old)) + (n_new - len(picked_new))
    if deficit > 0:
        leftover = old_hits[len(picked_old):] + new_hits[len(picked_new):]
        return picked_old + picked_new + leftover[:deficit]
    return picked_old + picked_new


TESTAMENT_LABELS = {"en": ("OT", "NT"), "fi": ("VT", "UT")}


def format_hits(hits, show_score=False, lang="en", show_testament=False):
    """lang controls display only: book names and verse text are looked
    up in that translation (see src/lang_lookup.py) while search/rerank
    always ran in English against the indexed text."""
    book_names = BOOK_NAMES_BY_LANG.get(lang, BOOK_NAMES_BY_LANG["en"])
    ot_label, nt_label = TESTAMENT_LABELS.get(lang, TESTAMENT_LABELS["en"])
    lines = []
    for h in hits:
        m = h["meta"]
        text = h["text"]
        if lang != "en":
            translated = localized_text(lang, m["book_num"], m["chapter"], m["verse_start"], m["verse_end"])
            if translated:
                text = translated
        book_name = book_names.get(m["book_num"], m["book"])
        ref = f"{book_name} {m['chapter']}:{m['verse_start']}-{m['verse_end']}"
        tags = []
        if show_testament:
            tags.append(ot_label if m["book_num"] <= OT_MAX_BOOK_NUM else nt_label)
        if show_score and "rerank_score" in h:
            tags.append(f"score={h['rerank_score']:.1f}")
        prefix = f"[{ref} | {' | '.join(tags)}]" if tags else f"[{ref}]"
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query or theme, e.g. 'God as healer'")
    parser.add_argument("-k", type=int, default=None, help="Max passages to keep (default: 8, or unlimited when --min-score is used)")
    parser.add_argument("--answer", action="store_true", help="Also ask the local LLM to synthesize a summary (default: just list the matched passages)")
    parser.add_argument("--rerank", action="store_true", help="Rerank candidates with the local LLM before keeping results")
    parser.add_argument("--fetch-k", type=int, default=None, help="Candidates to retrieve before reranking (default: 3x -k, or 60 in --min-score recall mode)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Recall mode (requires --rerank): keep every candidate scoring >= this (0-10) instead of a fixed top-k",
    )
    parser.add_argument("--lang", choices=["en", "fi"], default="en", help="Language of your query and the answer (retrieval always runs in English)")
    parser.add_argument(
        "--both-testaments",
        action="store_true",
        help="Query the Old and New Testament separately so both are represented, instead of one pooled search that can end up dominated by one testament",
    )
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

    if args.both_testaments:
        old_hits, new_hits = retrieve_both_testaments(search_query, fetch_k)
        hits = old_hits + new_hits
    else:
        hits = retrieve(search_query, fetch_k)

    if args.rerank:
        hits = rerank_hits(search_query, hits)
        if recall_mode:
            hits = [h for h in hits if h["rerank_score"] >= args.min_score]
            if args.k is not None:
                if args.both_testaments:
                    old_sorted, new_sorted = split_by_testament(hits)
                    hits = balance_take(old_sorted, new_sorted, args.k)
                else:
                    hits = hits[: args.k]
            label = f"--- Retrieved passages (reranked, score >= {args.min_score}, {len(hits)} found) ---"
        else:
            k = args.k or default_k
            if args.both_testaments:
                old_sorted, new_sorted = split_by_testament(hits)
                hits = balance_take(old_sorted, new_sorted, k)
            else:
                hits = hits[:k]
            label = "--- Retrieved passages (reranked) ---"
        print(label)
        print(format_hits(hits, show_score=True, lang=args.lang, show_testament=args.both_testaments))
    else:
        k = args.k or default_k
        if args.both_testaments:
            # each side already sorted by relevance (ascending distance) from its own query
            old_sorted, new_sorted = split_by_testament(hits)
            hits = balance_take(old_sorted, new_sorted, k)
        else:
            hits = hits[:k]
        print("--- Retrieved passages ---")
        print(format_hits(hits, lang=args.lang, show_testament=args.both_testaments))

    if args.answer:
        print("\n--- Answer ---")
        context = format_hits(hits, lang=args.lang)
        if args.lang == "fi":
            system_prompt = SYSTEM_PROMPT_FI
            user_prompt = (
                f"Aihe/kysymys: {args.query}\n\nRaamatunkohdat:\n{context}\n\n"
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
    try:
        main()
    except FileNotFoundError as e:
        raise SystemExit(f"Error: {e}")
