# Gradio Transcript App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local Gradio web UI over the existing `transcribe.py` CLI for reviewing, searching (keyword + semantic), viewing, and chatting with saved YouTube transcripts, with clickable timestamps that seek an embedded player — CLI unchanged.

**Architecture:** Three modules. `transcribe.py` keeps the fetch/save logic and gains path helpers + a return value. `library.py` holds all new data logic (scan, search, chunking, embeddings, chat, settings) as pure-ish functions with no Gradio import. `app.py` is Gradio wiring only. Data is re-read from disk per interaction (no index); semantic vectors are cached per video as `.npy` keyed on JSON mtime + embedding-model identity.

**Tech Stack:** Python 3.14, Gradio, numpy, `requests` (already present), `youtube-transcript-api` (already present), and `fastembed` (ONNX, no torch) **if it installs on 3.14 — otherwise API embeddings only**.

## Global Constraints

- **Python `<3.15`** (venv is 3.14.4) — `youtube-transcript-api` requires it. Every dependency must install on cp314 or be optional.
- **`youtube-transcript-api` 1.x instance API** only: `YouTubeTranscriptApi().fetch(video_id)`. The 0.x static API does not exist here.
- **No new heavy dependencies**: reuse `requests` for all HTTP (no `openai` package). No vector database. No torch (fastembed is ONNX).
- **Storage in user space**, never the repo. Resolution order everywhere: `--output-dir`/explicit arg → `YT_TRANSCRIBE_DIR` env → platform default (Windows `%APPDATA%\youtube-transcribe`; else `$XDG_DATA_HOME/youtube-transcribe` or `~/.local/share/youtube-transcribe`).
- **Security**: bind `127.0.0.1` only, never `share=True`/`0.0.0.0`. Write `config.json` at mode `0600`. Never log or echo the API key (masked UI field).
- **Embedding cache** keyed on JSON mtime **and** slugified model id; query embedded with the currently-configured model; only same-model vectors searched.
- **Tests never download the 130MB model** — use synthetic vectors / monkeypatched embedder.

---

### Task 1: Path helpers + `transcribe()` return value

**Files:**
- Modify: `transcribe.py` (add `default_data_dir()`, `config_dir()`; `transcribe()` returns the JSON path; CLI default `--output-dir`)
- Test: `test_app.py` (create)

**Interfaces:**
- Produces:
  - `transcribe.default_data_dir() -> Path` — resolves `YT_TRANSCRIBE_DIR` env then platform default; creates nothing.
  - `transcribe.config_dir() -> Path` — Windows: `%APPDATA%\youtube-transcribe`; else `$XDG_CONFIG_HOME/youtube-transcribe` or `~/.config/youtube-transcribe`.
  - `transcribe.transcribe(video_url: str, languages: list[str], output_dir: Path) -> Path` — now returns the written `.json` path.

- [ ] **Step 1: Write failing tests**

```python
# test_app.py
import os
from pathlib import Path
import importlib

import transcribe


def _reload(monkeypatch, **env):
    for k in ("YT_TRANSCRIBE_DIR", "APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_env_override_wins(monkeypatch, tmp_path):
    _reload(monkeypatch, YT_TRANSCRIBE_DIR=str(tmp_path / "custom"))
    assert transcribe.default_data_dir() == tmp_path / "custom"


def test_xdg_data_home(monkeypatch, tmp_path):
    _reload(monkeypatch, XDG_DATA_HOME=str(tmp_path / "xdg"))
    monkeypatch.setattr(transcribe.sys, "platform", "linux")
    assert transcribe.default_data_dir() == tmp_path / "xdg" / "youtube-transcribe"


def test_windows_appdata(monkeypatch, tmp_path):
    _reload(monkeypatch, APPDATA=str(tmp_path / "AppData"))
    monkeypatch.setattr(transcribe.sys, "platform", "win32")
    assert transcribe.default_data_dir() == tmp_path / "AppData" / "youtube-transcribe"


def test_config_dir_linux(monkeypatch, tmp_path):
    _reload(monkeypatch, XDG_CONFIG_HOME=str(tmp_path / "cfg"))
    monkeypatch.setattr(transcribe.sys, "platform", "linux")
    assert transcribe.config_dir() == tmp_path / "cfg" / "youtube-transcribe"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest test_app.py -v`
Expected: FAIL — `AttributeError: module 'transcribe' has no attribute 'default_data_dir'`

- [ ] **Step 3: Implement the helpers**

Add near the top of `transcribe.py` (after imports; `sys` and `Path` already imported; add `import os`):

```python
APP_DIR_NAME = "youtube-transcribe"


def default_data_dir() -> Path:
    """Where transcripts live: env override, else per-platform user data dir."""
    env = os.environ.get("YT_TRANSCRIBE_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def config_dir() -> Path:
    """Where config.json lives (same folder as data on Windows)."""
    if sys.platform == "win32":
        return default_data_dir()
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIR_NAME
```

- [ ] **Step 4: Make `transcribe()` return the JSON path**

At the end of `transcribe()` (currently ends after the `print("Done.")`), change the final lines to return `json_path`:

```python
    print(f"Saved:   {json_path}")
    print(f"Saved:   {md_path}")
    print("Done.")
    return json_path
```

- [ ] **Step 5: Switch CLI default output dir**

In `main()`, change the `--output-dir` argument default from the string `"transcripts"` to the resolved user-space dir:

```python
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Root output directory (default: per-platform user data dir)",
    )
```

Then after `args = parser.parse_args(argv)`:

```python
    output_dir = Path(args.output_dir) if args.output_dir else default_data_dir()
```

and pass `output_dir` into `transcribe(args.url, args.languages, output_dir)`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `venv/bin/python -m pytest test_app.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add transcribe.py test_app.py
git commit -m "feat: user-space data/config dirs + transcribe() returns json path"
```

---

### Task 2: Library scan + keyword search

**Files:**
- Create: `library.py`
- Test: `test_app.py` (append)

**Interfaces:**
- Consumes: `transcribe.default_data_dir`.
- Produces:
  - `library.format_timestamp(seconds: float) -> str` — `"m:ss"`, or `"h:mm:ss"` past an hour.
  - `library.scan_library(root: Path) -> list[dict]` — one dict per valid `.json`: keys `video_id, title, channel, channel_slug, published, language_code, is_generated, json_path (str), md_path (str), snippet_count`. Corrupt/unreadable JSON skipped (warn to stderr). Sorted by `published` desc then title.
  - `library.load_transcript(json_path: str) -> dict` — the full parsed JSON payload (includes `snippets`).
  - `library.keyword_search(root: Path, query: str, per_video_cap: int = 20, total_cap: int = 200) -> tuple[list[dict], bool]` — hit dicts `{video_id, title, channel, start (float), text, json_path}`; bool = truncated. Case-insensitive substring over each snippet's text.

- [ ] **Step 1: Write failing tests**

```python
# test_app.py (append)
import json
import library


def _write_video(root, channel, title, vid, snippets, published="2026-01-01"):
    slug = channel.lower().replace(" ", "-")
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title, "channel": channel, "channel_url": "",
        "published": published, "video_id": vid,
        "video_url": f"https://www.youtube.com/watch?v={vid}",
        "language": "English", "language_code": "en", "is_generated": True,
        "snippets": snippets,
    }
    p = d / f"{title.lower().replace(' ', '-')}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_format_timestamp():
    assert library.format_timestamp(0) == "0:00"
    assert library.format_timestamp(65) == "1:05"
    assert library.format_timestamp(3725) == "1:02:05"


def test_scan_skips_corrupt(tmp_path, capsys):
    _write_video(tmp_path, "Chan A", "Hello World", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    bad = tmp_path / "chan-a" / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    rows = library.scan_library(tmp_path)
    assert len(rows) == 1
    assert rows[0]["title"] == "Hello World"


def test_keyword_search_and_cap(tmp_path):
    snippets = [{"text": f"the word agent number {i}", "start": float(i), "duration": 1.0}
                for i in range(50)]
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa", snippets)
    hits, truncated = library.keyword_search(tmp_path, "AGENT", per_video_cap=20)
    assert len(hits) == 20
    assert truncated is True
    assert all("agent" in h["text"].lower() for h in hits)
    assert hits[0]["start"] == 0.0


def test_keyword_search_no_match(tmp_path):
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                 [{"text": "nothing here", "start": 0.0, "duration": 1.0}])
    hits, truncated = library.keyword_search(tmp_path, "zzz")
    assert hits == []
    assert truncated is False
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest test_app.py -v -k "timestamp or scan or keyword"`
Expected: FAIL — `ModuleNotFoundError: No module named 'library'`

- [ ] **Step 3: Implement `library.py`**

```python
"""Data logic for the transcript app: scan, search, chunking, embeddings, chat, settings.

No Gradio import here — this module stays UI-agnostic and unit-testable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


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
    rows.sort(key=lambda r: (r["published"], r["title"]), reverse=True)
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
```

- [ ] **Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest test_app.py -v -k "timestamp or scan or keyword"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add library.py test_app.py
git commit -m "feat: library scan + capped keyword search"
```

---

### Task 3: Settings load/save

**Files:**
- Modify: `library.py`
- Test: `test_app.py` (append)

**Interfaces:**
- Consumes: `transcribe.config_dir`.
- Produces:
  - `library.DEFAULT_SETTINGS: dict` — `{"embedding_provider": "local", "embedding_model": "BAAI/bge-small-en-v1.5", "api_base_url": "", "api_key": "", "chat_model": ""}`.
  - `library.settings_path() -> Path` — `config_dir() / "config.json"`.
  - `library.load_settings() -> dict` — defaults merged with file; missing file → defaults.
  - `library.save_settings(settings: dict) -> None` — writes JSON at mode `0600`, creating the dir.

- [ ] **Step 1: Write failing tests**

```python
# test_app.py (append)
import stat


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "config_dir", lambda: tmp_path)
    s = library.load_settings()
    assert s["embedding_provider"] == "local"       # default
    s["api_key"] = "secret"
    library.save_settings(s)
    assert library.load_settings()["api_key"] == "secret"


def test_settings_file_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "config_dir", lambda: tmp_path)
    library.save_settings(library.DEFAULT_SETTINGS.copy())
    mode = stat.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest test_app.py -v -k settings`
Expected: FAIL — `AttributeError: module 'library' has no attribute 'load_settings'`

- [ ] **Step 3: Implement**

Add to `library.py` (add `import os` and `from transcribe import config_dir` at top):

```python
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
            merged.update(json.loads(p.read_text(encoding="utf-8")))
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
```

- [ ] **Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest test_app.py -v -k settings`
Expected: PASS (Windows note: the permissions test may be skipped/relaxed on non-POSIX — acceptable, it asserts on the POSIX dev env.)

- [ ] **Step 5: Commit**

```bash
git add library.py test_app.py
git commit -m "feat: settings load/save with 0600 perms"
```

---

### Task 4: Chunking + embedding cache + semantic search

**Files:**
- Modify: `library.py`, `requirements.txt`
- Test: `test_app.py` (append)

**Interfaces:**
- Consumes: `scan_library`, `load_transcript`, `load_settings`, `format_timestamp`.
- Produces:
  - `library.chunk_snippets(snippets: list[dict], window_seconds: float = 45.0) -> list[dict]` — deterministic; each chunk `{text, start}` groups consecutive snippets until `window_seconds` elapse from the chunk's first start.
  - `library.embedding_model_id(settings: dict) -> str` — `f"{provider}:{model}"`.
  - `library.slugify_model(model_id: str) -> str` — filename-safe.
  - `library.embed_texts(texts: list[str], settings: dict) -> "np.ndarray"` — local (fastembed) or API; **injectable** via module global `_EMBED_FN` for tests.
  - `library.cache_path_for(json_path: str, model_id: str) -> Path` — `<json>.<slug>.npy`.
  - `library.embed_video(json_path: str, settings: dict) -> tuple["np.ndarray", list[dict]]` — returns (vectors[n_chunks, dim], chunks); reads cache if mtime+model match, else embeds and writes cache.
  - `library.semantic_search(root: Path, query: str, settings: dict, top_k: int = 20, only_json: str | None = None) -> list[dict]` — hit dicts identical in shape to keyword hits (`video_id, title, channel, start, text, json_path`) plus `score`. Lazily embeds any video missing a current-model cache.

**IMPORTANT — dependency verify-first:** before writing implementation, run:
`venv/bin/pip install fastembed && venv/bin/python -c "from fastembed import TextEmbedding; print('ok')"`
- If it prints `ok`: add `fastembed` to `requirements.txt`, local embeddings are the default.
- If it FAILS (no cp314 wheel for onnxruntime): **do not** add `fastembed`; leave the local path guarded by a `try/except ImportError` that raises a clear "local embeddings unavailable on this Python — configure an API embedding endpoint in Settings" error. The API path (Step 3 below) works regardless. Record which happened in the commit message.

- [ ] **Step 1: Write failing tests (synthetic embedder — never downloads the model)**

```python
# test_app.py (append)
import numpy as np


def test_chunk_snippets_windows():
    snips = [{"text": f"s{i}", "start": float(i * 10), "duration": 10.0} for i in range(10)]
    chunks = library.chunk_snippets(snips, window_seconds=45.0)
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["text"].startswith("s0")
    # first chunk spans starts 0,10,20,30,40 (<=45 from 0) -> next chunk starts at 50
    assert chunks[1]["start"] == 50.0


def test_semantic_search_ranks_by_cosine(tmp_path, monkeypatch):
    # Two snippets; fake embedder maps text -> fixed vectors so ranking is deterministic.
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa", [
        {"text": "cats and dogs", "start": 0.0, "duration": 40.0},
        {"text": "quantum physics", "start": 60.0, "duration": 40.0},
    ])
    vectors = {
        "cats and dogs": np.array([1.0, 0.0], dtype="float32"),
        "quantum physics": np.array([0.0, 1.0], dtype="float32"),
        "kittens": np.array([0.9, 0.1], dtype="float32"),   # query
    }

    def fake_embed(texts, settings):
        return np.vstack([vectors[t] for t in texts])

    monkeypatch.setattr(library, "_EMBED_FN", fake_embed)
    monkeypatch.setattr(library, "load_settings",
                        lambda: {**library.DEFAULT_SETTINGS})
    hits = library.semantic_search(tmp_path, "kittens",
                                   library.DEFAULT_SETTINGS, top_k=2)
    assert hits[0]["text"].startswith("cats")   # closest to the query vector
    assert "score" in hits[0]


def test_cache_invalidates_on_model_change(tmp_path, monkeypatch):
    jp = _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                      [{"text": "hello", "start": 0.0, "duration": 5.0}])
    calls = {"n": 0}

    def fake_embed(texts, settings):
        calls["n"] += 1
        return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setattr(library, "_EMBED_FN", fake_embed)
    s1 = {**library.DEFAULT_SETTINGS, "embedding_model": "model-a"}
    s2 = {**library.DEFAULT_SETTINGS, "embedding_model": "model-b"}
    library.embed_video(str(jp), s1)
    library.embed_video(str(jp), s1)   # cached, no new embed call
    assert calls["n"] == 1
    library.embed_video(str(jp), s2)   # different model -> re-embed
    assert calls["n"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest test_app.py -v -k "chunk or semantic or cache_invalidates"`
Expected: FAIL — `AttributeError: module 'library' has no attribute 'chunk_snippets'`

- [ ] **Step 3: Implement**

Add to `library.py` (`import re`, `import numpy as np` at top):

```python
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


def _default_embed(texts: list[str], settings: dict) -> "np.ndarray":
    if settings["embedding_provider"] == "api":
        return _api_embed(texts, settings)
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings unavailable on this Python — configure an API "
            "embedding endpoint in Settings."
        ) from exc
    model = TextEmbedding(model_name=settings["embedding_model"])
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
```

- [ ] **Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest test_app.py -v -k "chunk or semantic or cache_invalidates"`
Expected: PASS

- [ ] **Step 5: Update requirements.txt**

Add `numpy` (and `fastembed` only if Step 0 verify succeeded):

```
numpy
# fastembed   # add ONLY if it installs on this Python (see Task 4 verify-first)
```

- [ ] **Step 6: Commit**

```bash
git add library.py test_app.py requirements.txt
git commit -m "feat: chunking + model-keyed embedding cache + cosine semantic search"
```

---

### Task 5: Chat over retrieved chunks

**Files:**
- Modify: `library.py`
- Test: `test_app.py` (append)

**Interfaces:**
- Consumes: `semantic_search`, `format_timestamp`.
- Produces:
  - `library.build_chat_prompt(query: str, hits: list[dict]) -> list[dict]` — OpenAI-style messages; system message instructs citing as `[title @ mm:ss]`; user message embeds each hit as `"[<title> @ <mm:ss>] <text>"`.
  - `library.chat(query: str, root: Path, settings: dict, only_json: str | None = None, top_k: int = 8) -> str` — retrieves, builds prompt, POSTs to `<base>/chat/completions`, returns the answer string. Raises a clear error if `api_base_url`/`chat_model` unset. HTTP via `_CHAT_FN` injectable global for tests.

- [ ] **Step 1: Write failing tests**

```python
# test_app.py (append)
def test_build_chat_prompt_has_citation_instruction():
    hits = [{"title": "My Vid", "start": 65.0, "text": "hello world",
             "video_id": "x", "channel": "c", "json_path": "p"}]
    msgs = library.build_chat_prompt("what is said?", hits)
    assert msgs[0]["role"] == "system"
    assert "mm:ss" in msgs[0]["content"].lower() or "@" in msgs[0]["content"]
    assert "[My Vid @ 1:05]" in msgs[1]["content"]


def test_chat_calls_endpoint(tmp_path, monkeypatch):
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                 [{"text": "hello", "start": 0.0, "duration": 5.0}])
    monkeypatch.setattr(library, "semantic_search",
                        lambda *a, **k: [{"title": "Vid", "start": 0.0,
                                          "text": "hello", "video_id": "aaaaaaaaaaa",
                                          "channel": "Chan A", "json_path": "p"}])
    monkeypatch.setattr(library, "_CHAT_FN",
                        lambda messages, settings: "The answer [Vid @ 0:00]")
    s = {**library.DEFAULT_SETTINGS, "api_base_url": "http://x/v1", "chat_model": "m"}
    out = library.chat("q", tmp_path, s)
    assert "answer" in out


def test_chat_requires_config(tmp_path):
    try:
        library.chat("q", tmp_path, library.DEFAULT_SETTINGS)
        assert False, "expected error"
    except ValueError as e:
        assert "Settings" in str(e) or "endpoint" in str(e).lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest test_app.py -v -k chat`
Expected: FAIL — `AttributeError: module 'library' has no attribute 'build_chat_prompt'`

- [ ] **Step 3: Implement**

Add to `library.py`:

```python
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
    import requests
    resp = requests.post(
        settings["api_base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {settings['api_key']}"},
        json={"model": settings["chat_model"], "messages": messages},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat(query: str, root: Path, settings: dict, only_json: str | None = None,
         top_k: int = 8) -> str:
    if not settings.get("api_base_url") or not settings.get("chat_model"):
        raise ValueError("Chat is not configured — set an API endpoint and "
                         "chat model in Settings.")
    hits = semantic_search(root, query, settings, top_k=top_k, only_json=only_json)
    fn = _CHAT_FN or _default_chat
    return fn(build_chat_prompt(query, hits), settings)
```

- [ ] **Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest test_app.py -v -k chat`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `venv/bin/python -m pytest test_app.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add library.py test_app.py
git commit -m "feat: chat over retrieved chunks via OpenAI-compatible endpoint"
```

---

### Task 6: Gradio app (`app.py`)

**Files:**
- Create: `app.py`
- Modify: `requirements.txt` (add `gradio`)
- Test: manual smoke check (UI — no unit tests)

**Interfaces:**
- Consumes: everything in `library.py` + `transcribe.transcribe`, `transcribe.default_data_dir`.
- Produces: `app.py` runnable as `venv/bin/python app.py` launching Gradio on `127.0.0.1`.

**Design notes (build exactly this way — corrected mechanism):**
- **Player seek must NOT rely on `<script>` inside `gr.HTML`.** Browsers do not
  execute `<script>` inserted via innerHTML, which is how Gradio sets HTML
  content — so a script tag there is silently dead. Instead:
  - `ytSeek` is defined **once** via `demo.load(js=SEEK_JS)` (Gradio's `js=`
    hook runs on the client).
  - The player is a plain `<iframe id="yt-player" … enablejsapi=1>` — renders
    with no script. Selecting a video re-renders `player_html` with a fresh
    iframe (same id, new video id + `start`).
  - `ytSeek(sec)` posts a `seekTo` command to the iframe via `postMessage`.
    Each `[m:ss]` stamp is `<a onclick="ytSeek(N);return false">` — onclick
    attributes *do* fire when set via innerHTML (only `<script>` is blocked).
- **Library/hit table** is a `gr.Dataframe` with clean columns only (no file
  paths). Row-click reads the full dict from a `gr.State` list by row index —
  never from the visible cells. No `<mark>` in the table; highlight matches
  only inside the transcript HTML.
- **State**: one `visible_state` holds whatever dict-list the table currently
  shows (library rows or search hits); `current_json` holds the loaded video
  for chat "this video" scope.
- **Fetch** shows progress; **localhost bind**; **API key field masked**.

- [ ] **Step 1: Implement `app.py`**

```python
"""Gradio UI for browsing, searching, viewing, and chatting with transcripts.

Business logic lives in library.py; this file is wiring only.
"""
from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

import library
from transcribe import default_data_dir, transcribe

ROOT = default_data_dir()


LIB_HEADERS = ["Channel", "Title", "Published", "Lang"]
HIT_HEADERS = ["Channel", "Title", "Time", "Match"]

# Defined ONCE on the client via demo.load(js=…). Gradio's js= hook executes;
# a <script> inside gr.HTML would NOT. ytSeek posts a command to the iframe.
SEEK_JS = """
() => {
  window.ytCmd = function(func, args) {
    var f = document.getElementById('yt-player');
    if (f && f.contentWindow) {
      f.contentWindow.postMessage(
        JSON.stringify({event: 'command', func: func, args: args}), '*');
    }
  };
  window.ytSeek = function(sec) {
    window.ytCmd('seekTo', [sec, true]);
    window.ytCmd('playVideo', []);
  };
}
"""


def _lib_rows(rows: list[dict]) -> list[list[str]]:
    return [[r["channel"], r["title"], r["published"], r["language_code"]] for r in rows]


def _hit_rows(hits: list[dict]) -> list[list[str]]:
    return [[h["channel"], h["title"], library.format_timestamp(h["start"]),
             (h["text"][:120] + "…") if len(h["text"]) > 120 else h["text"]]
            for h in hits]


def load_library():
    rows = library.scan_library(ROOT)
    stats = f"{len(rows)} videos · {len({r['channel_slug'] for r in rows})} channels"
    return rows, gr.update(value=_lib_rows(rows), headers=LIB_HEADERS), stats


def _player_iframe(video_id: str, start: float) -> str:
    # ponytail: postMessage may no-op if clicked before the iframe's JS-API
    # handshake completes; fine for v1 — the stamp just needs a second click.
    return (f'<iframe id="yt-player" width="100%" height="360" frameborder="0" '
            f'allow="autoplay; encrypted-media" allowfullscreen '
            f'src="https://www.youtube.com/embed/{video_id}'
            f'?enablejsapi=1&start={int(start)}&autoplay=1"></iframe>')


def _transcript_html(data: dict, highlight: str = "") -> str:
    chunks = library.chunk_snippets(data.get("snippets", []))
    hl = highlight.strip().lower()
    parts = []
    for c in chunks:
        text = html.escape(c["text"])
        if hl and hl in c["text"].lower():
            text = f"<mark>{text}</mark>"
        stamp = library.format_timestamp(c["start"])
        parts.append(
            f'<p><a href="#" onclick="ytSeek({int(c["start"])});return false" '
            f'style="color:#ea580c;font-weight:600;text-decoration:none">'
            f'[{stamp}]</a> {text}</p>'
        )
    return "\n".join(parts)


def show_video(json_path: str, seek: float = 0.0, highlight: str = ""):
    data = library.load_transcript(json_path)
    player = (_player_iframe(data["video_id"], seek)
              + f'<div class="transcript">{_transcript_html(data, highlight)}</div>')
    meta = (f"**{data['title']}**  \n"
            f"Channel: {data['channel']} · Published: {data.get('published') or 'unknown'} "
            f"· Lang: {data['language_code']} · Auto-generated: "
            f"{'yes' if data['is_generated'] else 'no'}  \n"
            f"[Open on YouTube]({data['video_url']})")
    md = Path(json_path).with_suffix(".md")
    md_text = md.read_text(encoding="utf-8") if md.exists() else "_No .md file._"
    return player, meta, md_text, json_path


def do_search(query: str, mode: str):
    if not query.strip():
        rows = library.scan_library(ROOT)
        return gr.update(value=_lib_rows(rows), headers=LIB_HEADERS), "Full library", rows
    if mode == "Semantic":
        hits = library.semantic_search(ROOT, query, library.load_settings())
        truncated = False
    else:
        hits, truncated = library.keyword_search(ROOT, query)
    note = f"{len(hits)} matches" + (" (truncated)" if truncated else "")
    return gr.update(value=_hit_rows(hits), headers=HIT_HEADERS), note, hits


def on_row_select(evt: gr.SelectData, visible: list[dict], search_query: str):
    # Read the full dict from state by row index — never from the visible cells.
    d = visible[evt.index[0]]
    return show_video(d["json_path"], d.get("start", 0.0), search_query or "")


def do_fetch(url: str, progress=gr.Progress()):
    if not url.strip():
        return "Enter a URL.", *load_library()[1:]
    try:
        progress(0.3, desc="Fetching transcript…")
        transcribe(url, ["en"], ROOT)
        _, table, stats = load_library()
        return "Fetched.", table, stats
    except Exception as exc:  # noqa: BLE001 — surface any fetch failure to the UI
        _, table, stats = load_library()
        return f"Error: {exc}", table, stats


def do_chat(query: str, scope: str, current_json: str):
    try:
        only = current_json if scope == "This video" and current_json else None
        return library.chat(query, ROOT, library.load_settings(), only_json=only)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def save_settings_ui(provider, model, base, key, chat_model):
    library.save_settings({
        "embedding_provider": provider, "embedding_model": model,
        "api_base_url": base, "api_key": key, "chat_model": chat_model,
    })
    return "Saved."


with gr.Blocks(title="Transcript Library") as demo:
    visible_state = gr.State([])   # dict-list currently shown in the table
    current_json = gr.State("")

    stats_md = gr.Markdown("Loading…")
    with gr.Row():
        url_in = gr.Textbox(label="Add a video", placeholder="YouTube URL or ID", scale=4)
        fetch_btn = gr.Button("Fetch", scale=1)
    fetch_msg = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=5):
            search_in = gr.Textbox(label="Search", placeholder="Search transcripts…")
            mode = gr.Radio(["Keyword", "Semantic"], value="Keyword", label="Mode")
            search_note = gr.Markdown("")
            table = gr.Dataframe(headers=LIB_HEADERS, interactive=False, wrap=True)
        with gr.Column(scale=7):
            with gr.Tab("Viewer"):
                meta_md = gr.Markdown("")
                player_html = gr.HTML("")
            with gr.Tab("Markdown"):
                md_view = gr.Markdown("")
            with gr.Tab("Chat"):
                chat_scope = gr.Radio(["This video", "Whole library"],
                                      value="Whole library", label="Scope")
                chat_in = gr.Textbox(label="Ask")
                chat_btn = gr.Button("Send")
                chat_out = gr.Markdown("")
            with gr.Tab("Settings"):
                s = library.load_settings()
                prov = gr.Radio(["local", "api"], value=s["embedding_provider"],
                                label="Embedding provider")
                emodel = gr.Textbox(s["embedding_model"], label="Embedding model")
                base = gr.Textbox(s["api_base_url"], label="API base URL")
                key = gr.Textbox(s["api_key"], label="API key", type="password")
                cmodel = gr.Textbox(s["chat_model"], label="Chat model")
                save_btn = gr.Button("Save settings")
                save_msg = gr.Markdown("")

    # js=SEEK_JS defines window.ytSeek on the client at load (executes; a
    # <script> in gr.HTML would not).
    demo.load(load_library, outputs=[visible_state, table, stats_md], js=SEEK_JS)
    fetch_btn.click(do_fetch, [url_in], [fetch_msg, table, stats_md])
    search_in.submit(do_search, [search_in, mode], [table, search_note, visible_state])
    mode.change(do_search, [search_in, mode], [table, search_note, visible_state])
    table.select(on_row_select, [visible_state, search_in],
                 [player_html, meta_md, md_view, current_json])
    chat_btn.click(do_chat, [chat_in, chat_scope, current_json], [chat_out])
    save_btn.click(save_settings_ui, [prov, emodel, base, key, cmodel], [save_msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, show_api=False)
```

- [ ] **Step 2: Add gradio to requirements.txt**

```
gradio
```

- [ ] **Step 3: Install and smoke-test manually**

Run:
```bash
venv/bin/pip install -r requirements.txt
venv/bin/python app.py
```
Expected: browser opens at `http://127.0.0.1:7860`. Verify: library table lists existing videos; clicking a row loads the player + transcript; clicking a `[m:ss]` stamp seeks the video; keyword search returns hit rows and clicking one seeks; Markdown tab shows the `.md`; Settings saves. (Semantic/chat need the embedding stack / an API endpoint — test if available.)

- [ ] **Step 4: Commit**

```bash
git add app.py requirements.txt
git commit -m "feat: Gradio transcript app (browse, search, player-seek, chat, settings)"
```

---

### Task 7: Launchers + docs + gitignore

**Files:**
- Create: `app.sh`, `app.bat`
- Modify: `transcribe.sh`, `.gitignore` (create if absent), `CLAUDE.md`

**Interfaces:** none (scripts + docs).

- [ ] **Step 1: Write `app.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
  venv/bin/pip install -r requirements.txt
fi
exec venv/bin/python app.py "$@"
```

Run: `chmod +x app.sh`

- [ ] **Step 2: Write `app.bat`**

```bat
@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  python -m venv venv
  venv\Scripts\pip install -r requirements.txt
)
venv\Scripts\python app.py %*
```

- [ ] **Step 3: Add the same bootstrap to `transcribe.sh`**

Read the current `transcribe.sh`; ensure it creates the venv if missing before running, matching `app.sh`'s pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
  venv/bin/pip install -r requirements.txt
fi
exec venv/bin/python transcribe.py "$@"
```

- [ ] **Step 4: Create/update `.gitignore`**

```
venv/
transcripts/
__pycache__/
*.pyc
```

- [ ] **Step 5: Update `CLAUDE.md`**

Update these sections to match reality:
- **Commands**: add `./app.sh` (launches the web UI, bootstraps venv, opens browser).
- **Output layout**: transcripts now default to the user-space data dir (env `YT_TRANSCRIBE_DIR` → platform default), not `transcripts/` in the repo. Note the one-time migration: move any existing `transcripts/<channel>/…` into the new location (print it with `venv/bin/python -c "import transcribe; print(transcribe.default_data_dir())"`).
- **Architecture**: add `library.py` (data logic) and `app.py` (Gradio UI); note semantic search (fastembed/API), chat (OpenAI-compatible endpoint), and settings in `config.json`.

- [ ] **Step 6: Verify launchers work end-to-end**

Run: `./app.sh` (from a clean checkout ideally). Expected: venv exists or is created, browser opens, app works.

- [ ] **Step 7: Commit**

```bash
git add app.sh app.bat transcribe.sh .gitignore CLAUDE.md
git commit -m "feat: self-bootstrapping launchers + gitignore + docs update"
```

---

## Self-Review

**Spec coverage:**
- Storage/user-space + resolution order → Task 1 ✓
- `transcribe()` return value → Task 1 ✓
- Library scan (skip corrupt) → Task 2 ✓
- Keyword search + hit caps → Task 2 ✓
- Settings load/save + 0600 + masked key → Task 3 (perms), Task 6 (masked field) ✓
- Chunking (deterministic) → Task 4 ✓
- Embedding cache keyed on mtime + slugified model id → Task 4 ✓
- Local (fastembed) + API embeddings, verify-first wheel check → Task 4 ✓
- Semantic search (cosine, lazy embed) → Task 4 ✓
- Chat (retrieval + OpenAI-compatible, cite [title @ mm:ss], not-configured message) → Task 5 (+ UI Task 6) ✓
- Player seek mechanism (iframe `enablejsapi=1` + `ytSeek` via `demo.load(js=…)` + `postMessage` — NOT a `<script>` in gr.HTML) → Task 6 ✓
- Table = gr.Dataframe + .select() reading from `visible_state`, clean columns, highlight in viewer only → Task 6 ✓
- Chat citation cross-video seek + YouTube fallback → **partially** Task 6 (within-video ytSeek + "Open on YouTube" present; cross-video click-to-load from chat text is not wired because chat output is Markdown). See note below.
- Markdown tab → Task 6 ✓
- Localhost bind, no share → Task 6 ✓
- Launchers (bin vs Scripts) + transcribe.sh → Task 7 ✓
- gitignore venv + transcripts → Task 7 ✓
- Docs deliverable (CLAUDE.md/README) → Task 7 ✓
- Tests never download model → Task 4 (synthetic vectors) ✓

**Known simplification (flagged, not a gap):** The spec's cross-video chat-citation *click-to-load-and-seek* is simplified in v1 to: within-video seeks work via `ytSeek`, and every answer can be followed by selecting the cited video from the library + its transcript stamp; the always-works YouTube link is present on the viewer. Wiring clickable cross-video seek out of Markdown chat output requires a JS→Gradio bridge that isn't worth it for v1. `ponytail:` acceptable ceiling — revisit if chat becomes the primary navigation path.

**Placeholder scan:** none — every code step has complete code.

**Type consistency:** hit dicts share the shape `{video_id, title, channel, start, text, json_path}` (semantic adds `score`) across Tasks 2/4/5/6 ✓; `settings` dict keys consistent across Tasks 3/4/5/6 ✓; `_EMBED_FN`/`_CHAT_FN` injection points match tests ✓.
