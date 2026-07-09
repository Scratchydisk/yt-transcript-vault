# Gradio Transcript App — Design

**Date:** 2026-07-09
**Status:** Approved pending user review

## Purpose

Add a local web UI (Gradio) on top of the existing `transcribe.py` CLI for
reviewing, searching (keyword + semantic), viewing, and chatting with saved
YouTube transcripts, with clickable timestamps that seek an embedded player.
The CLI keeps working exactly as before. Mockup approved by user:
https://claude.ai/code/artifact/1fb0c855-7d11-45f2-9d68-3ca234db2c0f

## Files

| File | Role |
|------|------|
| `transcribe.py` | Existing fetch/save logic, unchanged — plus new `default_data_dir()`; CLI default `--output-dir` switches to it |
| `library.py` | New. All data logic: scan, keyword search, chunking, embeddings + cache, semantic search, chat retrieval/API calls, settings load/save |
| `app.py` | New. Gradio UI only — no business logic |
| `app.sh` / `app.bat` | Self-bootstrapping launchers |
| `test_app.py` | Tests for the pure functions |
| `requirements.txt` | + `gradio`, `fastembed`, `numpy` |

## Storage (housekeeping)

Transcripts move out of the repo into user space:

- Resolution order: `--output-dir` flag → `YT_TRANSCRIBE_DIR` env var →
  platform default.
- Platform default (stdlib only, no platformdirs):
  - Windows: `%APPDATA%\youtube-transcribe`
  - Else: `$XDG_DATA_HOME/youtube-transcribe` or `~/.local/share/youtube-transcribe`
- Same resolver shared by CLI and app.
- Migration: user moves the existing `transcripts/` folder manually (one-time,
  documented in README/CLAUDE.md). `transcripts/` added to `.gitignore` either way.
- Settings: `config.json` in `~/.config/youtube-transcribe` (respecting
  `$XDG_CONFIG_HOME`) on Linux/macOS; on Windows it lives in the same
  `%APPDATA%\youtube-transcribe` folder as the transcripts.

## UI layout (single page + tabs, per approved mockup)

1. **Header** — title, library stats, Refresh button.
2. **Add video** — URL box + Fetch button → calls existing `transcribe()`,
   surfaces success/error inline, refreshes library.
3. **Left panel: Library + Search**
   - Search box with **Keyword / Semantic** mode toggle.
   - Empty query → full library table (channel, title, published, language).
   - With query → hit rows (channel, title, timestamp, matching snippet).
     Both search modes return the same row shape.
   - Selecting any row loads the viewer; selecting a hit row seeks the player
     to the hit timestamp.
4. **Right panel: Viewer**
   - YouTube embed iframe (`youtube.com/embed/<id>?start=<sec>&autoplay=1`).
   - Title + metadata chips (channel, published, language, auto-generated,
     "Open on YouTube" external link).
   - **Transcript tab**: paragraphs built from timed snippets, each prefixed
     with a clickable `[mm:ss]` stamp that reloads the iframe at that offset.
     Search hits highlighted.
   - **Markdown tab**: the `.md` file rendered, with its filesystem path shown.
5. **Chat tab** — scope dropdown (this video / whole library), question box,
   answer with `[title @ mm:ss]` citations rendered as the same seek-links.
   No conversation persistence in v1.
6. **Settings tab** — embedding provider (local / API), OpenAI-compatible
   base URL, API key, chat model name, embedding model name. API embedding
   mode calls the endpoint's `/embeddings` route; local mode ignores the
   endpoint entirely. Saved to `config.json`. Defaults work with zero
   configuration (local embeddings, chat disabled until an endpoint is set).

## Search

- **Keyword**: case-insensitive substring scan over JSON snippets in memory,
  re-read from disk per query (picks up CLI-fetched videos automatically).
  No index. Fine to ~hundreds of videos.
- **Semantic**: snippets grouped into ~45-second windows (chunk = text +
  start time). Embedded with **fastembed** (`BAAI/bge-small-en-v1.5`, ONNX —
  no torch). Vectors cached per video as `.npy` beside the JSON, invalidated
  by JSON mtime. Query embedding → numpy cosine similarity over all cached
  vectors → top-k hit rows.
- Model download (~130MB) happens lazily on first semantic search, with a
  progress message — not at app startup.
- Explicitly **no vector database**. Upgrade path if the library ever reaches
  tens of thousands of videos: sqlite-vec. Not before.

## Chat

- Retrieval: semantic top-k chunks (scoped to one video or whole library).
- Generation: OpenAI-compatible `POST /chat/completions` via the configured
  base URL + key + model. One code path covers OpenAI, OpenRouter, Ollama,
  LM Studio. Implemented with `requests` (already a dependency) — the
  `openai` package is not needed for one endpoint.
- Prompt: retrieved chunks with video title + timestamp, instruction to cite
  as `[title @ mm:ss]`; citations post-processed into seek-links.
- If no chat endpoint configured, the Chat tab shows a pointer to Settings
  instead of erroring.

## Startup

- `./app.sh` (POSIX) and `app.bat` (Windows): create `venv/` and
  `pip install -r requirements.txt` if missing, then run `app.py`.
- `app.py` launches Gradio with `inbrowser=True` → browser opens itself.
- Fresh-user path: clone → `./app.sh` → browser tab. No manual steps.
- `transcribe.sh` gains the same bootstrap check.

## Error handling

- Fetch errors reuse the exceptions `main()` already catches
  (`ValueError`, `TranscriptsDisabled`, `VideoUnavailable`,
  `CouldNotRetrieveTranscript`) and surface as inline UI messages.
- Corrupt/missing JSON files are skipped in the library scan with a warning.
- Chat/embedding API failures show the error message in the UI; never crash
  the app.

## Testing

`test_app.py` (plain pytest, no fixtures): data-dir resolution per platform
(env-var monkeypatching), chunking windows, keyword search, cosine ranking,
timestamp formatting, settings round-trip. Network, LLM calls, and the UI
itself: manual smoke check.

## Explicitly out of scope (v1)

- Vector database (numpy brute force instead)
- Browser-side models / transformers.js (server-side local model instead)
- Transcript editing, chat history persistence, pagination, auth
- Karaoke-style live highlighting synced to playback
