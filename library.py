"""Data logic for the transcript app: scan, search, chunking, embeddings, chat, settings.

No Gradio import here — this module stays UI-agnostic and unit-testable.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from transcribe import config_dir


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def load_transcript(json_path: str) -> dict:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def scan_library(root: Path) -> list[dict]:
    root = Path(root)
    rows: list[dict] = []
    for jp in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"  ! Skipping unreadable transcript {jp} ({exc}).", file=sys.stderr)
            continue
        rows.append({
            "video_id": data.get("video_id", ""),
            "title": data.get("title", jp.stem),
            "channel": data.get("channel", ""),
            "channel_slug": jp.parent.name,
            "published": data.get("published", ""),
            "language_code": data.get("language_code", ""),
            "is_generated": data.get("is_generated", False),
            "json_path": str(jp),
            "md_path": str(jp.with_suffix(".md")),
            "snippet_count": len(data.get("snippets", [])),
        })
    rows.sort(key=lambda r: r["title"])
    rows.sort(key=lambda r: r["published"], reverse=True)
    return rows


def keyword_search(root: Path, query: str, per_video_cap: int = 20,
                   total_cap: int = 200) -> tuple[list[dict], bool]:
    query = query.strip().lower()
    if not query:
        return [], False
    hits: list[dict] = []
    truncated = False
    for row in scan_library(root):
        data = load_transcript(row["json_path"])
        per_video = 0
        for snip in data.get("snippets", []):
            text = snip.get("text", "")
            if query in text.lower():
                if per_video >= per_video_cap or len(hits) >= total_cap:
                    truncated = True
                    break
                hits.append({
                    "video_id": row["video_id"], "title": row["title"],
                    "channel": row["channel"], "start": float(snip.get("start", 0.0)),
                    "text": text, "json_path": row["json_path"],
                })
                per_video += 1
        if len(hits) >= total_cap:
            truncated = True
            break
    return hits, truncated


# The local fastembed model that ships by default and is always available
# (no API, no key) — always offered as an embedding choice.
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Default base URL prefilled when the user picks the Ollama provider.
OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_SETTINGS = {
    "embedding_provider": "local",              # "local" (fastembed) | "api" (OpenAI-compatible)
    "embedding_model": LOCAL_EMBED_MODEL,
    "api_type": "openai",                       # "openai" | "ollama" — picks the chat endpoint
    "api_base_url": "",                         # OpenAI-compatible base, e.g. https://api.openai.com/v1
    "api_key": "",
    "chat_model": "",
    "num_ctx": 0,                               # Ollama context window; 0 = don't send (server default)
}


def settings_path() -> Path:
    return config_dir() / "config.json"


def load_settings() -> dict:
    merged = DEFAULT_SETTINGS.copy()
    p = settings_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
        except (ValueError, OSError):
            pass
    return merged


def save_settings(settings: dict) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass  # ponytail: best-effort on filesystems without POSIX perms (Windows)


_EMBED_FN = None  # tests inject a fake; production uses _default_embed


def chunk_snippets(snippets: list[dict], window_seconds: float = 45.0) -> list[dict]:
    chunks: list[dict] = []
    cur_texts: list[str] = []
    cur_start: float | None = None
    for snip in snippets:
        start = float(snip.get("start", 0.0))
        text = snip.get("text", "").strip()
        if not text:
            continue
        if cur_start is None:
            cur_start = start
        if start - cur_start > window_seconds and cur_texts:
            chunks.append({"text": " ".join(cur_texts), "start": cur_start})
            cur_texts, cur_start = [], start
        cur_texts.append(text)
    if cur_texts:
        chunks.append({"text": " ".join(cur_texts), "start": cur_start or 0.0})
    return chunks


def embedding_model_id(settings: dict) -> str:
    return f"{settings['embedding_provider']}:{settings['embedding_model']}"


def slugify_model(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def cache_path_for(json_path: str, model_id: str) -> Path:
    p = Path(json_path)
    return p.with_suffix(f".{slugify_model(model_id)}.npy")


_LOCAL_MODELS: dict[str, object] = {}  # cache: model_name -> TextEmbedding instance


def _get_local_model(model_name: str):
    if model_name not in _LOCAL_MODELS:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings unavailable on this Python — configure an API "
                "embedding endpoint in Settings."
            ) from exc
        _LOCAL_MODELS[model_name] = TextEmbedding(model_name=model_name)
    return _LOCAL_MODELS[model_name]


def _default_embed(texts: list[str], settings: dict) -> "np.ndarray":
    # Anything other than "local" (openai / ollama / legacy "api") goes through
    # the OpenAI-compatible /embeddings endpoint.
    if settings["embedding_provider"] != "local":
        return _api_embed(texts, settings)
    model = _get_local_model(settings["embedding_model"])
    return np.array(list(model.embed(texts)), dtype="float32")


def _api_embed(texts: list[str], settings: dict) -> "np.ndarray":
    import requests
    resp = requests.post(
        settings["api_base_url"].rstrip("/") + "/embeddings",
        headers={"Authorization": f"Bearer {settings['api_key']}"},
        json={"model": settings["embedding_model"], "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return np.array([d["embedding"] for d in data], dtype="float32")


def embed_texts(texts: list[str], settings: dict) -> "np.ndarray":
    fn = _EMBED_FN or _default_embed
    return fn(texts, settings)


_MODELS_FN = None  # tests inject a fake; production uses _default_model_ids


def _default_model_ids(settings: dict) -> list[str]:
    """OpenAI-compatible GET <base>/models -> sorted list of model ids."""
    import requests
    headers = {}
    if settings.get("api_key"):
        headers["Authorization"] = f"Bearer {settings['api_key']}"
    resp = requests.get(
        settings["api_base_url"].rstrip("/") + "/models",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))


def discover_models(settings: dict) -> dict:
    """Discover selectable models from the configured API connection.

    Returns {"embedding": [...], "chat": [...]}. The local fastembed model is
    ALWAYS offered for embeddings, independent of the API. If an api_base_url
    is set, its /models list is added (used for chat, and available for API
    embeddings). Discovery keys off the API connection, NOT the embedding
    source, so local-embeddings + API-chat discovers chat models fine.
    """
    embedding = [LOCAL_EMBED_MODEL]
    chat: list[str] = []
    if settings.get("api_base_url"):
        fn = _MODELS_FN or _default_model_ids
        ids = fn(settings)
        chat = list(ids)
        embedding += [m for m in ids if m != LOCAL_EMBED_MODEL]
    return {"embedding": embedding, "chat": chat}


def embed_video(json_path: str, settings: dict) -> tuple["np.ndarray", list[dict]]:
    data = load_transcript(json_path)
    chunks = chunk_snippets(data.get("snippets", []))
    model_id = embedding_model_id(settings)
    cache = cache_path_for(json_path, model_id)
    src_mtime = Path(json_path).stat().st_mtime
    if cache.exists() and cache.stat().st_mtime >= src_mtime:
        return np.load(cache), chunks
    if not chunks:
        vectors = np.zeros((0, 1), dtype="float32")
    else:
        vectors = embed_texts([c["text"] for c in chunks], settings)
    np.save(cache, vectors)
    return vectors, chunks


def semantic_search(root: Path, query: str, settings: dict, top_k: int = 20,
                    only_json: str | None = None) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    qvec = embed_texts([query], settings)[0]
    qnorm = np.linalg.norm(qvec) or 1.0
    rows = scan_library(root)
    if only_json:
        rows = [r for r in rows if r["json_path"] == only_json]
    scored: list[dict] = []
    for row in rows:
        vectors, chunks = embed_video(row["json_path"], settings)
        if len(chunks) == 0 or vectors.shape[0] != len(chunks):
            continue
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0
        sims = (vectors @ qvec) / (norms * qnorm)
        for i, chunk in enumerate(chunks):
            scored.append({
                "video_id": row["video_id"], "title": row["title"],
                "channel": row["channel"], "start": chunk["start"],
                "text": chunk["text"], "json_path": row["json_path"],
                "score": float(sims[i]),
            })
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:top_k]


_CHAT_FN = None  # tests inject; production uses _default_chat

_CHAT_SYSTEM = (
    "You answer questions about YouTube video transcripts. Use only the "
    "provided excerpts. Cite every claim inline as [title @ mm:ss] using the "
    "titles and timestamps shown. If the excerpts don't cover it, say so."
)


def build_chat_prompt(query: str, hits: list[dict]) -> list[dict]:
    lines = [f"[{h['title']} @ {format_timestamp(h['start'])}] {h['text']}"
             for h in hits]
    context = "\n\n".join(lines) if lines else "(no relevant excerpts found)"
    return [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}"},
    ]


def _default_chat(messages: list[dict], settings: dict) -> str:
    if settings.get("api_type") == "ollama":
        return _ollama_chat(messages, settings)
    import requests
    payload = {"model": settings["chat_model"], "messages": messages}
    resp = requests.post(
        settings["api_base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {settings['api_key']}"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _ollama_chat(messages: list[dict], settings: dict) -> str:
    # Ollama's OpenAI-compatible /v1 endpoint silently drops `options` (so
    # num_ctx never takes effect there). Its native /api/chat honours them, so
    # we call that directly when the API type is Ollama.
    import requests
    # Settings store the /v1 base (used for embeddings + model discovery); the
    # native API lives at the host root under /api/chat.
    base = settings["api_base_url"].rstrip("/")
    if base.endswith("/v1"):
        base = base[:-len("/v1")]
    payload = {
        "model": settings["chat_model"],
        "messages": messages,
        "stream": False,
    }
    num_ctx = settings.get("num_ctx") or 0
    if num_ctx:
        payload["options"] = {"num_ctx": int(num_ctx)}
    # Ollama needs no auth; forward a key only if one is set (e.g. behind a proxy).
    headers = {"Authorization": f"Bearer {settings['api_key']}"} \
        if settings.get("api_key") else {}
    resp = requests.post(base.rstrip("/") + "/api/chat", headers=headers,
                         json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


_CITE_RE = re.compile(r"\[([^\]@]+?)\s*@\s*(\d+):(\d{2})(?::(\d{2}))?\]")


def link_citations(text: str, hits: list[dict]) -> str:
    """Turn `[title @ mm:ss]` citations into YouTube deep links (new tab)."""
    by_title = {h["title"].strip().lower(): h["video_id"] for h in hits}

    def repl(m: re.Match) -> str:
        vid = by_title.get(m.group(1).strip().lower())
        if not vid:
            return m.group(0)  # unknown title — leave as plain text
        parts = [int(g) for g in m.groups()[1:] if g is not None]
        secs = parts[0] * 60 + parts[1] if len(parts) == 2 else \
            parts[0] * 3600 + parts[1] * 60 + parts[2]
        url = f"https://www.youtube.com/watch?v={vid}&t={secs}s"
        return f"[{m.group(0)}]({url})"

    return _CITE_RE.sub(repl, text)


def chat(query: str, root: Path, settings: dict, only_json: str | None = None,
         top_k: int = 8) -> str:
    if not settings.get("api_base_url") or not settings.get("chat_model"):
        raise ValueError("Chat is not configured — set an API endpoint and "
                         "chat model in Settings.")
    hits = semantic_search(root, query, settings, top_k=top_k, only_json=only_json)
    fn = _CHAT_FN or _default_chat
    return link_citations(fn(build_chat_prompt(query, hits), settings), hits)
