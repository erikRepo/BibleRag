import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3.5:9b")
BIBLE_XML = os.environ.get("BIBLE_XML", "./data/EnglishKJBible.xml")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./data/chroma")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "4"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "1"))
BIBLE_REPO_PATH = os.path.expanduser(
    os.environ.get("BIBLE_REPO_PATH", "~/git/Holy-Bible-XML-Format")
)
