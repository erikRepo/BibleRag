"""Translate the search query into English for retrieval, since the
indexed Bible text and embedding model are English. Plain text in,
plain text out - no JSON needed for a single string.

think=False is used, but a bare zero-shot prompt turned out unreliable
for short Finnish theological phrases (e.g. "Jumala parantajana" ->
"God's teacher" instead of "God as healer") - and think=True doesn't
fix it either, since qwen3.5 can spend 2000+ reasoning tokens
deliberating over a two-word phrase and never reach an answer before
hitting num_predict. A few-shot prompt fixes accuracy while staying
fast (no thinking needed)."""

from .ollama_client import chat

TRANSLATE_SYSTEM = (
    "You are a translation engine for a Bible search tool. Translate the "
    "input into a short, natural English Bible-search query. Output ONLY "
    "the English text, nothing else - no quotes, no notes, no alternatives. "
    "If the input is already in English, repeat it unchanged.\n\n"
    "Examples:\n"
    "Finnish: Jumala parantajana\nEnglish: God as healer\n\n"
    "Finnish: hyva paimen\nEnglish: the Good Shepherd\n\n"
    "Finnish: synti ja anteeksianto\nEnglish: sin and forgiveness"
)


def to_english(text: str) -> str:
    return chat(TRANSLATE_SYSTEM, text, think=False).strip()
