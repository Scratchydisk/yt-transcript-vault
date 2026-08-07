# yt-transcript-vault

Fetch YouTube video transcripts to disk, then browse, search, and chat with them
from a local web UI.

Given a video URL it saves the transcript as raw JSON **and** AI-ready Markdown,
organised by channel. A [Gradio](https://www.gradio.app/) app sits on top for
keyword + semantic search, a timestamp-seekable player, and chatting with your
transcripts via any OpenAI-compatible LLM.

## Quick start

```bash
# Web UI — bootstraps a venv on first run, then opens your browser
./app.sh

# Transcribe a single video from the CLI
./transcribe.sh "https://www.youtube.com/watch?v=VIDEO_ID"
./transcribe.sh "https://youtu.be/VIDEO_ID" --languages en de   # language preference order
```

Both scripts create the `venv/` and install `requirements.txt` on first run.
Windows users have `app.bat`. To run the Python entry points directly:

```bash
venv/bin/python app.py            # web UI
venv/bin/python transcribe.py "<url>"
```

Requires Python **3.10–3.14** (`youtube-transcript-api` needs `<3.15`).

### Serving it on your network

The UI binds `127.0.0.1` and opens a browser — it is a personal tool, and a
fresh clone stays off the network. To run it headless on a machine you reach
from elsewhere on a **trusted** LAN, set `YT_BIND`:

```bash
YT_BIND=0.0.0.0 venv/bin/python app.py    # serve the LAN; skips the browser launch
```

There is no authentication in front of the UI, so anyone who can reach the port
can read your transcripts and change your settings. Keep it to networks you
trust. `YT_BIND` is a LAN bind and nothing more — this app never uses Gradio's
`share=True`, which would tunnel it to the public internet.

## Features

- **Transcribe** — combines `youtube-transcript-api`, YouTube oEmbed (title/channel),
  and a watch-page scrape (publish date). No API key needed.
- **Keyword search** across your whole library.
- **Semantic search** — embeddings via local `fastembed` (`BAAI/bge-small-en-v1.5`)
  or an OpenAI-compatible API, cached per file.
- **Viewer** with a YouTube player that seeks to any `[m:ss]` transcript stamp.
- **Chat** — retrieves the most relevant chunks and answers via an
  OpenAI-compatible endpoint, citing sources as `[title @ mm:ss]`. Each citation
  is a clickable link that opens the video at that timestamp in a new tab.
- **Transfer** — export your entire library as a portable zip (optionally with
  settings), or selectively import videos from another user's export. Perfect for
  moving your vault between PCs or sharing curated collections.

## Where transcripts are stored

By default, in your platform user-data directory (not the repo):

| OS | Location |
|----|----------|
| Linux/macOS | `~/.local/share/youtube-transcribe/` |
| Windows | `%APPDATA%\youtube-transcribe\` |

Override with the `YT_TRANSCRIBE_DIR` environment variable. Layout:

```
<data-dir>/<channel-slug>/<video-title>.json   # metadata + timed snippets
<data-dir>/<channel-slug>/<video-title>.md      # metadata header + paragraphed transcript
```

## Transfer & Backup

The **Transfer** tab (in the web UI) lets you export and import your library:

**Export.** Click "Export library" to download a zip of all your transcripts. You
can optionally include `config.json` (API keys, model settings) — off by default
so shared zips don't leak credentials.

**Import.** Upload a zip from another user's export. The UI shows which videos are
new vs. already in your library, and you can selectively import by channel or
individual video. Existing videos are skipped (no overwrite).

## Configuration

Settings (API endpoints, model names, embedding provider) are edited in the UI's
**Settings** tab and saved to `config.json` with `0600` permissions. `config.json`
is git-ignored — your API keys stay local.

## Notes

- YouTube may return `RequestBlocked`/`IpBlocked` under heavy use (IP-based);
  `youtube-transcript-api` supports proxies if needed.
- Exit codes: `2` unparseable URL, `1` transcript unavailable, `0` success.

## License

[MIT](LICENSE) © Stewart McSporran
