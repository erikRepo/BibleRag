# BibleRag

A small, fully local Retrieval-Augmented-Generation (RAG) system for
thematic Bible search — e.g. asking "where does the Bible describe God as
a healer?" and getting back grounded verses plus a synthesized answer,
with everything running on your own machine.

No API keys, no cloud services. The Bible text, the embeddings, and the
answer-generating LLM all run locally via [Ollama](https://ollama.com).

## How it works

1. `src/ingest.py` parses a Bible XML file (book → chapter → verse), chunks
   it into small sliding windows of verses, embeds each chunk locally, and
   stores the embeddings in a local [Chroma](https://www.trychroma.com/)
   vector database on disk.
2. `src/query.py` embeds your question/theme the same way, retrieves the
   most semantically similar passages from Chroma, and (optionally) asks a
   local LLM to summarize what those passages say, with citations.

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

Ask a question:

```bash
python -m src.query "God as healer"
python -m src.query "the Good Shepherd" -k 10
python -m src.query "sin and forgiveness" --no-answer   # just show passages, skip LLM synthesis
python -m src.query "God as healer" --rerank            # rerank candidates with the local LLM first
```

`--rerank` fetches a larger candidate pool from the vector store and asks
the local chat model to score each one's relevance before keeping the
top `-k`. It measurably improves thematic search quality for modest extra
latency (a few seconds) — see [RERANKING.md](RERANKING.md) for a worked
before/after comparison and why it stays fast.

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

## Notes / possible next steps

- Swap in a different translation by pointing `BIBLE_XML` at another file
  in the same XML shape and re-running `python -m src.ingest`.
- Re-ingesting with a different `CHUNK_SIZE`/`CHUNK_OVERLAP` just
  upserts new chunk IDs into the same collection; delete `data/chroma/`
  first if you want a clean rebuild.
- A cross-encoder reranking step after retrieval would improve precision
  further on fuzzy thematic queries — not implemented here.
