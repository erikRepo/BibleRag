"""Look up verse text in another language for display, using a local
clone of the Holy-Bible-XML-Format repo (BIBLE_REPO_PATH) and the
code -> filename mapping in languages.toml.

Search/embedding/reranking always run against the English index
(config.BIBLE_XML) - this module only swaps in another translation's
text for the passages that were already found, by book/chapter/verse
number, so results stay grounded in the same engine regardless of
display language.
"""

import os
import tomllib
import xml.etree.ElementTree as ET

from . import config

_LANGUAGES_TOML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "languages.toml")

_verse_cache: dict[str, dict[tuple[int, int, int], str]] = {}


def _load_language_map() -> dict[str, str]:
    with open(_LANGUAGES_TOML, "rb") as f:
        return tomllib.load(f)["languages"]


def _load_verses(lang: str) -> dict[tuple[int, int, int], str]:
    if lang in _verse_cache:
        return _verse_cache[lang]

    filename = _load_language_map().get(lang)
    if not filename:
        raise ValueError(f"No Bible file configured for language '{lang}' in languages.toml")

    path = os.path.join(config.BIBLE_REPO_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Bible file for language '{lang}' not found at {path}. "
            f"Check BIBLE_REPO_PATH in .env (currently {config.BIBLE_REPO_PATH})."
        )

    root = ET.parse(path).getroot()
    verses: dict[tuple[int, int, int], str] = {}
    for testament in root.findall("testament"):
        for book in testament.findall("book"):
            book_num = int(book.get("number"))
            for chapter in book.findall("chapter"):
                chapter_num = int(chapter.get("number"))
                for verse in chapter.findall("verse"):
                    verses[(book_num, chapter_num, int(verse.get("number")))] = (
                        verse.text or ""
                    ).strip()

    _verse_cache[lang] = verses
    return verses


def localized_text(lang: str, book_num: int, chapter: int, verse_start: int, verse_end: int) -> str | None:
    """Returns the verse range's text in the given language, or None if
    any part of the range is missing from that translation."""
    verses = _load_verses(lang)
    parts = []
    for v in range(verse_start, verse_end + 1):
        text = verses.get((book_num, chapter, v))
        if text is None:
            return None
        parts.append(f"{v}. {text}")
    return " ".join(parts)
