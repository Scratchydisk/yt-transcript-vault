# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Given a YouTube video URL, fetch its transcript and save it to disk as both raw JSON
and AI-ready Markdown, organized by channel. A Gradio web UI (`app.py`) sits on top for
browsing, keyword + semantic search, viewing transcripts with a timestamp-seekable
player, and chatting with the transcripts via an OpenAI-compatible LLM.

## Commands

```bash
# Launch the web UI (bootstraps venv if needed, opens browser)
./app.sh

# Transcribe a video via CLI (bootstraps venv if needed)
./transcribe.sh "https://www.youtube.com/watch?v=VIDEO_ID"
./transcribe.sh "https://youtu.be/VIDEO_ID" --languages en de   # language preference order

# Run Python entry points directly (equivalent)
venv/bin/python app.py                                          # web UI
venv/bin/python transcribe.py "<url>"                           # CLI transcriber

# Tests (pytest is not in requirements.txt: venv/bin/pip install pytest)
venv/bin/python -m pytest
venv/bin/python -m pytest -k notes                              # single area
```

Tests live in `test_app.py` (pytest, `tmp_path`-based, no network). No linter is configured.

## Output layout

By default, transcripts are stored in the platform-specific user data directory (see `default_data_dir()` in `transcribe.py`), not in the repo:

- **Linux/macOS**: `~/.local/share/youtube-transcribe/` (XDG_DATA_HOME)
- **Windows**: `%APPDATA%\youtube-transcribe\`
- **Override**: set `YT_TRANSCRIBE_DIR` environment variable

Structure:
```
<data-dir>/<channel-slug>/<video-title>.json      # metadata + raw timed snippets
<data-dir>/<channel-slug>/<video-title>.md         # metadata header + transcript grouped into paragraphs
<data-dir>/<channel-slug>/<video-title>.notes.md   # user notes (optional; included in search + chat)
<data-dir>/<channel-slug>/<video-title>.chat.md    # exported chat Q&As (optional; NOT searched)
```

- Channel folder is slugified (`"Dark Finance"` → `dark-finance`).
- File name is the sanitized video title (illegal filesystem chars stripped, capped at 120 chars).

**Migration:** if you have existing transcripts in `transcripts/`, move them to the user data dir:
```bash
mv transcripts/* "$(venv/bin/python -c 'import transcribe; print(transcribe.default_data_dir())')"/
```

## Architecture

### Modules

- **`transcribe.py`**: CLI entry point. Fetches transcripts from YouTube and saves as JSON + Markdown.
  - **Three data sources** combined (none alone is sufficient):
    - `youtube-transcript-api` (1.x, instance API: `YouTubeTranscriptApi().fetch(video_id)`) — snippets + language metadata, no title/channel.
    - YouTube **oEmbed** endpoint (`youtube.com/oembed`, via `requests`, no API key) — title, author (channel).
    - **Watch page HTML** scrape — publish date (via `datePublished` meta tag).
  - **Flow**: `extract_video_id` → `fetch_metadata` (oEmbed + fallbacks) → `fetch_publish_date` → fetch transcript (any available language if preferred not found) → `paragraphize` → write JSON + Markdown.
  - **Exit codes**: `2` = unparseable URL; `1` = transcript unavailable; `0` = success.

- **`library.py`**: Data logic for the UI. Scan transcripts, keyword search (capped at 20 hits per video / 200 total), chunking, embeddings, semantic search, chat, settings.
  - **Semantic search**: embeddings via `fastembed` (local `BAAI/bge-small-en-v1.5`) or OpenAI-compatible API, cached per-file with mtime+model-id key.
  - **Chat**: retrieves top-K chunks, calls the OpenAI-compatible endpoint (SSE)
    or Ollama's native `/api/chat` (NDJSON) and **streams** the reply; cites
    sources as `[title @ mm:ss]`. Reasoning ("thinking") is streamed into a
    collapsible block above the answer when the `think` setting is on.
  - **Settings** (`config.json`): API endpoints, model names, user preferences. Saved with `0600` mode, sensitive keys are masked in UI output.

- **`app.py`**: Gradio web UI. Runs on 127.0.0.1 (localhost, no share), opens browser.
  - **Layout**: left panel = "Add a video" box + a single search box with a Keyword/Semantic mode toggle (radio) over a results/library table; right panel = tabs **Viewer / Markdown / Notes / Chat / Settings / Transfer**.
  - **Player**: YouTube iframe (`enablejsapi=1`); the `ytSeek(seconds)` JS is defined once via `demo.load(js=...)` and each `[m:ss]` transcript stamp calls it via `postMessage` (seeking the currently-loaded video). Selecting a search hit loads that video — including a different one — cued (not autoplaying) at the hit timestamp.
  - **Notes**: per-video markdown saved to `<title>.notes.md`; included in keyword search (line hits, `[note]` prefix, 0:00) and in embeddings/chat (`notes_chunks`, cache invalidated by notes mtime + a chunk-count shape guard).
  - **Chat export**: "Copy to clipboard" (client-side JS reading a hidden textbox that mirrors the raw Q&A markdown) and "Save with video" (appends to `<title>.chat.md`, `---`-separated; never searched — search/chat only read `.json` + `.notes.md`).
  - **Settings**: embedding provider (local/API), OpenAI-compatible base URL + API key (masked) + embedding/chat model names; API type (OpenAI/Ollama); num_ctx (Ollama); "Stream reasoning (think)" toggle; saved to `config.json`.

### Launchers

- **`app.sh` / `app.bat`**: Self-bootstrapping — creates venv and installs requirements on first run, then launches the UI.
- **`transcribe.sh`**: Self-bootstrapping — creates venv if missing, then runs `transcribe.py`.

## Gotchas

- Pin matters: this targets **youtube-transcript-api 1.x**. The 0.x static API
  (`YouTubeTranscriptApi.get_transcript(...)`) does NOT exist here — use the instance API.
- `youtube-transcript-api` requires Python `<3.15`; the venv is on 3.14.4.
- YouTube may return `RequestBlocked`/`IpBlocked` on heavy use (IP-based); the library
  supports proxies if that becomes an issue.
