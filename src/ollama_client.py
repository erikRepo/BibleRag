import requests

from . import config


def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_many(texts: list[str]) -> list[list[float]]:
    return [embed(t) for t in texts]


def chat(system: str, user: str, think: bool = True) -> str:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": think,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
