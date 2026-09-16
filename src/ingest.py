"""Parse a Holy-Bible-XML-Format XML file, chunk it into overlapping
verse windows, embed each chunk with Ollama, and store it in a local
Chroma vector store."""

import xml.etree.ElementTree as ET

import chromadb

from . import config
from .books import BOOK_NAMES
from .ollama_client import embed


def load_verses(xml_path: str):
    root = ET.parse(xml_path).getroot()
    verses = []
    for testament in root.findall("testament"):
        for book in testament.findall("book"):
            book_num = int(book.get("number"))
            book_name = BOOK_NAMES.get(book_num, f"Book {book_num}")
            for chapter in book.findall("chapter"):
                chapter_num = int(chapter.get("number"))
                for verse in chapter.findall("verse"):
                    verses.append(
                        {
                            "book": book_name,
                            "book_num": book_num,
                            "chapter": chapter_num,
                            "verse": int(verse.get("number")),
                            "text": (verse.text or "").strip(),
                        }
                    )
    return verses


def chunk_verses(verses, size: int, overlap: int):
    """Slide a window over verses within the same book+chapter so a chunk
    never spans a chapter boundary."""
    chunks = []
    step = max(size - overlap, 1)
    by_chapter = {}
    for v in verses:
        by_chapter.setdefault((v["book_num"], v["chapter"]), []).append(v)

    for (book_num, chapter_num), vs in by_chapter.items():
        for i in range(0, len(vs), step):
            window = vs[i : i + size]
            if not window:
                continue
            text = " ".join(f"{w['verse']}. {w['text']}" for w in window)
            chunks.append(
                {
                    "id": f"{book_num}-{chapter_num}-{window[0]['verse']}-{window[-1]['verse']}",
                    "book": window[0]["book"],
                    "chapter": chapter_num,
                    "verse_start": window[0]["verse"],
                    "verse_end": window[-1]["verse"],
                    "text": text,
                }
            )
            if i + size >= len(vs):
                break
    return chunks


def main():
    print(f"Loading verses from {config.BIBLE_XML} ...")
    verses = load_verses(config.BIBLE_XML)
    print(f"Loaded {len(verses)} verses.")

    chunks = chunk_verses(verses, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Built {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection("bible")

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = [embed(c["text"]) for c in batch]
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "book": c["book"],
                    "chapter": c["chapter"],
                    "verse_start": c["verse_start"],
                    "verse_end": c["verse_end"],
                }
                for c in batch
            ],
        )
        print(f"Indexed {min(start + batch_size, len(chunks))}/{len(chunks)}")

    print("Done. Vector store written to", config.CHROMA_DIR)


if __name__ == "__main__":
    main()
