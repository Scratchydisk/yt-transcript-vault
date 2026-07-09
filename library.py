"""Data logic for the transcript app: scan, search, chunking, embeddings, chat, settings.

No Gradio import here — this module stays UI-agnostic and unit-testable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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


DEFAULT_SETTINGS = {
    "embedding_provider": "local",              # "local" | "api"
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "api_base_url": "",                         # OpenAI-compatible base, e.g. http://localhost:11434/v1
    "api_key": "",
    "chat_model": "",
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
