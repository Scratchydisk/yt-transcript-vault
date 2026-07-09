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

    # Two separate load handlers: one pure-JS to define window.ytSeek on the
    # client (executes; a <script> in gr.HTML would not), one for the data.
    # Kept separate so the no-return js can't clobber the data fn's outputs.
    demo.load(None, None, None, js=SEEK_JS)
    demo.load(load_library, outputs=[visible_state, table, stats_md])
    fetch_btn.click(do_fetch, [url_in], [fetch_msg, table, stats_md])
    search_in.submit(do_search, [search_in, mode], [table, search_note, visible_state])
    mode.change(do_search, [search_in, mode], [table, search_note, visible_state])
    table.select(on_row_select, [visible_state, search_in],
                 [player_html, meta_md, md_view, current_json])
    chat_btn.click(do_chat, [chat_in, chat_scope, current_json], [chat_out])
    save_btn.click(save_settings_ui, [prov, emodel, base, key, cmodel], [save_msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
