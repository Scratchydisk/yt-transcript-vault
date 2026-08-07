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
| `transcribe.py` | Existing fetch/save logic + new `default_data_dir()`; `transcribe()` gains a return value (the JSON path) so the app can select the new file — CLI behaviour otherwise unchanged; CLI default `--output-dir` switches to `default_data_dir()` |
| `library.py` | New. All data logic: scan, keyword search, chunking, embeddings + cache, semantic search, chat retrieval/API calls, settings load/save |
| `app.py` | New. Gradio UI only — no business logic |
| `app.sh` / `app.bat` | Self-bootstrapping launchers |
| `test_app.py` | Tests for the pure functions |
| `requirements.txt` | + `gradio`, `numpy`, and `fastembed` **if** it installs on this interpreter (see dependency risk below) |

### Dependency risk — verify before building semantic search

This repo is pinned to **Python 3.14.4** (`youtube-transcript-api` requires
`<3.15`). `fastembed` depends on `onnxruntime`, which historically lags new
Python releases. **The plan's first semantic-search task must verify a
`cp314` wheel installs** (`pip install fastembed` in the venv). If it does
not:
- Local embeddings are unavailable on this interpreter. The app still ships
  keyword search + viewer + chat; semantic search and library-wide chat
  retrieval require the **API embedding provider** (Settings → `/embeddings`
  route), which needs no local wheel.
- The Settings tab surfaces this clearly ("local embeddings unavailable on
  this Python; configure an API embedding endpoint") rather than crashing.
This keeps the app fully functional regardless of wheel availability.

## Storage (housekeeping)

Transcripts move out of the repo into user space:

- Resolution order: `--output-dir` flag → `YT_TRANSCRIBE_DIR` env var →
  platform default.
- Platform default (stdlib only, no platformdirs):
  - Windows: `%APPDATA%\youtube-transcribe`
  - Else: `$XDG_DATA_HOME/youtube-transcribe` or `~/.local/share/youtube-transcribe`
- Same resolver shared by CLI and app.
- Migration: user moves the existing `transcripts/` folder manually (one-time,
  documented in README/CLAUDE.md). `transcripts/` and `venv/` added to
  `.gitignore` either way.
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
   - **Table rendering**: rows are plain text in a `gr.Dataframe` with its
     `.select()` event for row-click (Gradio's Dataframe does not render HTML,
     so there is no `<mark>` highlight *in the table* — the mockup's in-table
     highlight is aspirational; v1 highlights matches only in the transcript
     viewer).
   - **Hit cap**: keyword search caps hits per video and overall (e.g. 20 per
     video, 200 total) so a common word can't return thousands of rows; the
     count line notes when results were truncated.
4. **Right panel: Viewer**
   - YouTube embed iframe (`youtube.com/embed/<id>?start=<sec>&autoplay=1`).
   - Title + metadata chips (channel, published, language, auto-generated,
     "Open on YouTube" external link).
   - **Transcript tab**: paragraphs built from timed snippets, each prefixed
     with a clickable `[mm:ss]` stamp that seeks the player to that offset.
     Search hits highlighted.

   **Player seek mechanism (load-bearing — must be built this way).**
   Gradio cannot attach click handlers to arbitrary HTML spans, and an
   `<a>` inside one component cannot reload an iframe in another without a
   server round-trip. So the player and the transcript live in a **single
   `gr.HTML` block** that loads the **YouTube IFrame Player API**
   (`<script src="https://www.youtube.com/iframe_api">`), instantiates a
   `YT.Player`, and exposes a `ytSeek(seconds)` JS function. Each `[mm:ss]`
   stamp is `<a href="#" onclick="ytSeek(123); return false">`, so seeking
   is entirely client-side — no Gradio event, no reload, instant. Selecting
   a different video re-renders this HTML block (new video ID) via a normal
   Gradio event. Note: this is the one place the app relies on youtube.com
   being reachable (as the embed already does).
   - **Markdown tab**: the `.md` file rendered, with its filesystem path shown.
5. **Chat tab** — scope dropdown (this video / whole library), question box,
   answer with `[title @ mm:ss]` citations. **Citation behaviour**: a
   citation for the currently-loaded video uses the client-side `ytSeek()`;
   a citation for any other video is wired to a Gradio event that loads that
   video into the viewer and seeks to the timestamp. Every citation also
   carries a plain `youtube.com/watch?v=…&t=…s` link as an always-works
   fallback. No conversation persistence in v1.
6. **Settings tab** — embedding provider (local / API), OpenAI-compatible
   base URL, API key, chat model name, embedding model name. API embedding
   mode calls the endpoint's `/embeddings` route; local mode ignores the
   endpoint entirely. Saved to `config.json`, written with **user-only
   permissions (0600)**; the API key is **never logged or echoed** in the UI
   (masked field). Defaults work with zero configuration (local embeddings,
   chat disabled until an endpoint is set).

## Search

- **Keyword**: case-insensitive substring scan over JSON snippets in memory,
  re-read from disk per query (picks up CLI-fetched videos automatically).
  No index. Fine to ~hundreds of videos.
- **Semantic**: snippets grouped into ~45-second windows (chunk = text +
  start time). Chunking is **deterministic from the JSON**, so only the
  vectors are cached — chunk text and start times are regenerated on load
  and mapped to vector rows by position (no separate metadata file).
  Embedded with **fastembed** (`BAAI/bge-small-en-v1.5`, ONNX — no torch).
- **Cache correctness**: vectors cached per video as `.npy` beside the JSON.
  The cache is keyed on **both** the JSON mtime **and the embedding model
  identity** (provider + model name, slugified — the model id contains `/`
  which is not filename-safe), stored in a small sidecar or the npy
  filename. Switching embedding model (e.g. local 384-dim bge-small → an API
  model of different dimension) invalidates the cache — otherwise cosine
  similarity would compare vectors from different spaces and return garbage.
  The query is always embedded with the currently-configured model, and only
  vectors from that same model are searched.
- Query embedding → numpy cosine similarity over matching cached vectors →
  top-k hit rows.
- Model download (~130MB) happens lazily on first semantic search, with a
  progress message — not at app startup.
- Explicitly **no vector database**. Upgrade path if the library ever reaches
  tens of thousands of videos: sqlite-vec. Not before.

## Chat

- Retrieval: semantic top-k chunks (scoped to one video or whole library).
  Whole-library scope and semantic search **lazily embed any video that
  lacks a current-model cache** before querying, showing progress — a video
  fetched by the CLI is embedded on first semantic use, not at fetch time.
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
- **Bind localhost by default** (`server_name="127.0.0.1"`) — a fresh clone is a
  personal tool and must not expose transcripts or the API key to the network
  without the user asking for it.
- **Never `share=True`.** That is Gradio's public internet tunnel via
  `gradio.live`, and nothing in this app should reach the open internet.
- Binding wider is an explicit opt-in via the `YT_BIND` environment variable
  (amended 2026-08-07): `YT_BIND=0.0.0.0` serves a headless deployment on a
  trusted LAN, and suppresses `inbrowser` since such a host has no browser.
  Unset means localhost, so the safe default survives.
- Launchers must use the platform venv path: `venv/bin/…` on POSIX,
  `venv\Scripts\…` in `app.bat`.
- Fresh-user path: clone → `./app.sh` → browser tab. No manual steps.
- `transcribe.sh` gains the same bootstrap check.
- In-app fetch is slow (network); show a `gr.Progress`/spinner so the UI
  doesn't look frozen. Autoplay after a timestamp click works because the
  click is a user gesture; the initial player load may start muted/paused,
  which is fine.

## Error handling

- Fetch errors reuse the exceptions `main()` already catches
  (`ValueError`, `TranscriptsDisabled`, `VideoUnavailable`,
  `CouldNotRetrieveTranscript`) and surface as inline UI messages.
- Corrupt/missing JSON files are skipped in the library scan with a warning.
- Chat/embedding API failures show the error message in the UI; never crash
  the app.

## Testing

`test_app.py` (plain pytest, no fixtures): data-dir resolution per platform
(env-var monkeypatching), chunking windows, keyword search, cosine ranking
(**using synthetic hand-written vectors — never downloads the 130MB model**),
cache-key invalidation on model change, timestamp formatting, settings
round-trip. Network, LLM calls, the real embedder, and the UI itself: manual
smoke check.

## Documentation deliverable

CLAUDE.md currently documents the in-repo `transcripts/` layout and the CLI
only. The plan **must** include updating it (and adding a short README section)
to reflect: the new user-space storage location + resolution order, the
`./app.sh` launch path, and the semantic/chat/settings features. Stale docs
pointing at the old location would misdirect the user about where transcripts
now live.

## Explicitly out of scope (v1)

- Vector database (numpy brute force instead)
- Browser-side models / transformers.js (server-side local model instead)
- Transcript editing, chat history persistence, pagination, auth
- Karaoke-style live highlighting synced to playback
