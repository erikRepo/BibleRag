# BibleRag

A small, fully local Retrieval-Augmented-Generation (RAG) system for
thematic Bible search — e.g. asking "where does the Bible describe God as
a healer?" and getting back a list of the matching verses, with everything
running on your own machine.

No API keys, no cloud services. The Bible text, the embeddings, and the
optional answer-generating LLM all run locally via [Ollama](https://ollama.com).

## How it works

1. `src/ingest.py` parses a Bible XML file (book → chapter → verse), chunks
   it into small sliding windows of verses, embeds each chunk locally, and
   stores the embeddings in a local [Chroma](https://www.trychroma.com/)
   vector database on disk.
2. `src/query.py` embeds your question/theme the same way and retrieves the
   most semantically similar passages from Chroma. By default it just
   lists the matched verses — no LLM commentary or interpretation, just
   "these are the passages, here's what they say." Add `--answer` if you
   also want a local LLM to write a cited summary on top.

Because retrieval is semantic (not keyword search), a query like
"God as healer" surfaces passages that never contain the word "healer" —
e.g. Exodus 15:26 or Isaiah 53:5.

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running locally.
  This project does *not* install Ollama for you — install it yourself
  from https://ollama.com (or your package manager), then make sure the
  `ollama` command works and the service is running (`ollama serve` /
  the desktop app).
- Two Ollama models pulled locally:
  ```bash
  ollama pull nomic-embed-text   # embedding model
  ollama pull qwen3.5:9b         # or any chat model you prefer
  ```
- A Bible text in the XML format used by
  [Holy-Bible-XML-Format](https://github.com/) (`<bible><testament><book><chapter><verse>`),
  book numbers 1-66 in standard order. A public-domain translation such as
  the King James Version works well for local LLMs (simpler, unambiguous
  English compared to modern paraphrases).
- Optional, only needed for `--lang`: a local clone of that same
  [Holy-Bible-XML-Format](https://github.com/) repo, which ships dozens of
  translations in other languages using the identical XML layout. Point
  `BIBLE_REPO_PATH` at it (see `.env.example`).

## Setup

```bash
git clone <this-repo>
cd BibleRag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env if you want different models, chunk sizes, or file paths

# put your Bible XML file where .env points (default: ./data/EnglishKJBible.xml)
cp /path/to/EnglishKJBible.xml ./data/
```

## Usage

Build the vector index (only needed once per Bible file / chunking config):

```bash
python -m src.ingest
```

Ask a question — by default this just lists the matched passages:

```bash
python -m src.query "God as healer"
python -m src.query "the Good Shepherd" -k 10
python -m src.query "God as healer" --rerank             # rerank candidates with the local LLM first
python -m src.query "God as healer" --rerank --min-score 7   # recall mode: every hit, not just top-k
python -m src.query "Jumala parantajana" --lang fi --rerank -k 5   # search in Finnish
python -m src.query "God as healer" --rerank --answer    # also ask the LLM for a cited summary
```

### Asking in another language

The Bible text and embedding model here are English-only, so `--lang fi`
translates your Finnish query to English before searching — retrieval and
reranking always run in English against the indexed text (`BIBLE_XML`).
The *displayed* verse text and book names, though, are looked up by
book/chapter/verse number in an actual Finnish translation (default:
`FinnishBible.xml`, the public-domain 1933/38 Kirkkoraamattu) from your
`BIBLE_REPO_PATH` clone — so what you read is a real Finnish translation,
not a machine translation of the English verse. Combined with `--answer`,
the chat model also writes its summary in Finnish, quoting that same
Finnish text.

`languages.toml` maps each `--lang` code to a filename inside
`BIBLE_REPO_PATH`; add a language by cloning
[Holy-Bible-XML-Format](https://github.com/) yourself (it's not vendored
into this repo — dozens of translations, no need to duplicate them all
here) and adding a line, e.g. `de = "GermanSCH2000Bible.xml"`.

Translating the *query itself* needed a few-shot prompt to be reliable
for short theological phrases — see the comments in `src/translate.py`
and `src/ollama_client.py` if you want the details (short version: a
thinking model can either mistranslate two-word phrases with thinking
off, or burn its whole token budget "thinking" about them with thinking
on and never answer — few-shot examples fixed it without needing
`think=True`).

`--rerank` fetches a larger candidate pool from the vector store and asks
the local chat model to score each one's relevance before keeping the
top `-k`. It measurably improves thematic search quality for modest extra
latency (a few seconds) — see [RERANKING.md](RERANKING.md) for a worked
before/after comparison and why it stays fast.

By default `-k` caps results to a fixed count (8), which is fine when you
want "the best few hits" but hides how many places a theme actually
appears. Add `--min-score N` (0-10, requires `--rerank`) to switch into
**recall mode**: instead of a fixed top-k, every candidate scoring `N` or
higher is kept — so a broad theme like "healing" can return 30+ passages
instead of being capped at 8. Recall mode also widens the initial vector
search pool (to 60 candidates by default; override with `--fetch-k`) so
there's more to find relevant hits among in the first place. You can still
combine it with `-k` to cap the recall-mode output if it returns too many.

## Configuration

All configuration lives in `.env` (copy from `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_HOST` | URL of your local Ollama server | `http://localhost:11434` |
| `EMBED_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `CHAT_MODEL` | Ollama chat model for answer synthesis | `qwen3.5:9b` |
| `BIBLE_XML` | Path to the source XML file | `./data/EnglishKJBible.xml` |
| `CHROMA_DIR` | Where the vector store is persisted | `./data/chroma` |
| `CHUNK_SIZE` | Verses per chunk | `4` |
| `CHUNK_OVERLAP` | Verse overlap between consecutive chunks | `1` |
| `BIBLE_REPO_PATH` | Local clone of Holy-Bible-XML-Format, for `--lang` display translations | `~/git/Holy-Bible-XML-Format` |

## Notes / possible next steps

- Swap in a different translation by pointing `BIBLE_XML` at another file
  in the same XML shape and re-running `python -m src.ingest`.
- Re-ingesting with a different `CHUNK_SIZE`/`CHUNK_OVERLAP` just
  upserts new chunk IDs into the same collection; delete `data/chroma/`
  first if you want a clean rebuild.
- A cross-encoder reranking step after retrieval would improve precision
  further on fuzzy thematic queries — not implemented here.
