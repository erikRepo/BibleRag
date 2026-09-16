# Reranking: does it help, and what does it cost?

Yes — for fuzzy, thematic queries it visibly improves ranking quality, and
because it's implemented as a lightweight local-LLM scoring pass (not a
loaded cross-encoder model) it stays cheap: ~2-3 seconds for 18 candidates.

## How it works

`src/rerank.py` takes the top `fetch_k` (default: `3 × k`) candidates from
the Chroma vector search and asks the local chat model (`qwen3.5:9b` via
Ollama) to score each one's relevance to the query on a 0-10 scale, then
keeps the top `k` by that score. This substitutes for a dedicated
cross-encoder model (e.g. `bge-reranker`) so the project doesn't need an
extra `torch`/`sentence-transformers` dependency — everything still runs
through the one Ollama chat model already required for answer synthesis.

**Important detail:** `qwen3.5:9b` is a reasoning model. With its default
"thinking" mode on, scoring a single passage took **over 60 seconds**
(the model reasons at length before emitting a number) — reranking 18
candidates would have taken minutes. Passing `"think": false` in the
Ollama `/api/chat` request drops this to **~150ms per passage**, with
no loss in scoring quality for this simple task. `src/ollama_client.chat`
now exposes a `think` parameter; reranking always calls it with
`think=False`, while final answer synthesis still uses the default
(`think=True`) since longer reasoning helps there.

Usage: add `--rerank` to `python -m src.query`.

## Comparison 1: "God as healer" (k=6)

**Vector search only** (`python -m src.query "God as healer" -k 6`):

1. Matthew 15:28-31 — general healing miracles, deaf/lame/blind
2. James 5:13-16 — anointing with oil, prayer for the sick
3. Matthew 8:4-7 — centurion's servant
4. **Psalms 68:19-22** — praise psalm, "God of our salvation"; only tangentially about healing
5. **Titus 3:4-7** — salvation/regeneration language, not physical healing
6. John 11:1-4 — setup verses for the Lazarus story (before any healing happens)

**With `--rerank`** (fetches 18 candidates, keeps top 6 by LLM relevance score):

1. James 5:13-16 — score 10.0
2. Luke 13:10-13 — score 10.0 (woman "loosed from infirmity", new)
3. Luke 14:1-4 — score 10.0 (healing on the sabbath, new)
4. Matthew 8:13-16 — score 10.0 (centurion's servant *and* Peter's mother-in-law's fever, new)
5. Matthew 15:28-31 — score 9.0
6. Matthew 8:4-7 — score 9.0

**What changed:** Psalms 68 and Titus 3 — both retrieved because they share
salvation/deliverance vocabulary with "healer" in embedding space, but
neither is actually about physical or spiritual healing — dropped out.
They were replaced by three new passages (Luke 13, Luke 14, Matthew 8:13-16)
that are unambiguous healing narratives the plain vector search hadn't
surfaced in the original top-6 at all (they were sitting further down
among the 18 candidates). John 11:1-4 also dropped despite being
Lazarus-adjacent, since verses 1-4 alone don't describe healing.

## Comparison 2: "the Good Shepherd" (k=5)

**Vector search only:**

1. John 10:13-16
2. John 10:10-13
3. Zechariah 11:16-17 — a *bad* shepherd oracle (idol shepherd), thematically inverted
4. **Luke 2:7-10** — nativity scene, shepherds "abiding in the field"; keyword overlap ("shepherd") but no thematic connection to God-as-shepherd
5. Psalms 23:1-4

**With `--rerank`:**

1. John 10:13-16 — score 10.0
2. John 10:10-13 — score 10.0
3. Psalms 23:1-4 — score 10.0
4. Ezekiel 34:4-7 — score 10.0 (new — God rebuking false shepherds, promising to shepherd Israel himself)
5. Zechariah 11:16-17 — score 9.0

**What changed:** Luke 2's nativity shepherds — retrieved purely because
"shepherd" appears in the text, with zero connection to the "Good Shepherd"
theme — were correctly dropped and replaced by Ezekiel 34, a much more
substantive passage about God as shepherd. Psalms 23 also moved up from
5th to 3rd.

## Takeaways

- Plain vector search over short verse windows reliably finds the "obvious"
  passages (John 10, Psalms 23) but is prone to keyword-coincidence noise
  (Luke 2's shepherds) and semantic near-misses (Psalms 68 / Titus 3 sharing
  "salvation" vocabulary with "healer" without being about healing).
- Reranking consistently pushed the coincidental/near-miss hits out and
  surfaced more thematically precise passages that vector search alone
  had ranked lower than the top-k cutoff.
- Cost is small once `think=False` is used: ~150ms/passage, ~2-3s total
  for a `k=6`/`fetch_k=18` query — cheap enough to leave on by default for
  thematic searches. Skip `--rerank` only if you want the lowest possible
  latency or are doing simple keyword-adjacent lookups where the top
  vector hits are already unambiguous.
