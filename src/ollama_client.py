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


def chat(system: str, user: str, think: bool = True, num_predict: int = 4096) -> str:
    """num_predict defaults generously because thinking models spend a
    chunk of the token budget on <think> reasoning before any answer
    text - too low a default silently truncates the response to an
    empty string once the model runs out of budget mid-thought."""
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
            "options": {"num_predict": num_predict},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
