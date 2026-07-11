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
# Per-view column widths: library wants a wide Title; search wants a wide Match
# (the snippet), or it wraps to one word per line and rows balloon.
LIB_WIDTHS = ["22%", "48%", "16%", "14%"]
HIT_WIDTHS = ["15%", "26%", "9%", "50%"]

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
             (h["text"][:90] + "…") if len(h["text"]) > 90 else h["text"]]
            for h in hits]


def load_library():
    rows = library.scan_library(ROOT)
    stats = f"{len(rows)} videos · {len({r['channel_slug'] for r in rows})} channels"
    return rows, gr.update(value=_lib_rows(rows), headers=LIB_HEADERS,
                           column_widths=LIB_WIDTHS), stats


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
        return (gr.update(value=_lib_rows(rows), headers=LIB_HEADERS,
                          column_widths=LIB_WIDTHS), "Full library", rows)
    if mode == "Semantic":
        hits = library.semantic_search(ROOT, query, library.load_settings())
        truncated = False
    else:
        hits, truncated = library.keyword_search(ROOT, query)
    note = f"{len(hits)} matches" + (" (truncated)" if truncated else "")
    return (gr.update(value=_hit_rows(hits), headers=HIT_HEADERS,
                      column_widths=HIT_WIDTHS), note, hits)


def on_row_select(evt: gr.SelectData, visible: list[dict], search_query: str):
    # Read the full dict from state by row index — never from the visible cells.
    d = visible[evt.index[0]]
    return show_video(d["json_path"], d.get("start", 0.0), search_query or "")


def do_fetch(url: str, progress=gr.Progress()):
    # Always return the refreshed rows into visible_state too, or row-select
    # after a fetch would index a stale list (wrong video / IndexError).
    if not url.strip():
        rows, table, stats = load_library()
        return "Enter a URL.", table, stats, rows
    try:
        progress(0.3, desc="Fetching transcript…")
        transcribe(url, ["en"], ROOT)
        rows, table, stats = load_library()
        return "Fetched.", table, stats, rows
    except Exception as exc:  # noqa: BLE001 — surface any fetch failure to the UI
        rows, table, stats = load_library()
        return f"Error: {exc}", table, stats, rows


def render_chat(thinking: str, answer: str) -> str:
    # <details> is open while only reasoning has arrived, then collapses once the
    # answer starts, so the reader lands on the answer with reasoning tucked away.
    if not thinking and not answer:
        return "_Thinking…_"
    parts = []
    if thinking:
        open_attr = "" if answer else " open"
        parts.append(f"<details{open_attr}><summary>Thinking…</summary>\n\n"
                     f"{thinking}\n\n</details>")
    if answer:
        parts.append(answer)
    return "\n\n".join(parts)


def do_chat(query: str, scope: str, current_json: str):
    # Generator so the "Thinking…" state renders immediately, before the (slow)
    # retrieval + first token — otherwise the UI looks frozen until done.
    yield "_Thinking…_"
    try:
        only = current_json if scope == "This video" and current_json else None
        streamed = False
        for thinking, answer in library.chat_stream(
                query, ROOT, library.load_settings(), only_json=only):
            streamed = True
            yield render_chat(thinking, answer)
        if not streamed:
            yield "_(no response)_"
    except Exception as exc:  # noqa: BLE001
        yield f"Error: {exc}"


def save_settings_ui(provider, model, api_type, base, key, chat_model, num_ctx, think):
    library.save_settings({
        "embedding_provider": provider, "embedding_model": model,
        "api_type": api_type, "api_base_url": base, "api_key": key,
        "chat_model": chat_model, "num_ctx": int(num_ctx or 0),
        "think": bool(think),
    })
    return "Saved."


def on_api_type_change(api_type: str, base: str):
    # The API-type radio only prefills the base URL; it doesn't affect embeddings.
    if api_type == "ollama" and not base.strip():
        return library.OLLAMA_BASE_URL
    return base


def discover_models_ui(base, key):
    settings = {"api_base_url": base, "api_key": key}
    try:
        found = library.discover_models(settings)
    except Exception as exc:  # noqa: BLE001 — surface discovery failure in the UI
        return (gr.update(choices=[library.LOCAL_EMBED_MODEL]), gr.update(),
                f"Discovery failed: {exc}")
    emb, chat = found["embedding"], found["chat"]
    note = (f"Found {len(chat)} model(s)." if chat
            else f"Local only — {library.LOCAL_EMBED_MODEL} available.")
    return gr.update(choices=emb), gr.update(choices=chat), note


# Compact the library/search table so all columns fit the narrow left panel
# instead of scrolling off — smaller font, tighter cell padding.
TABLE_CSS = """
/* Gradio 6 renders body cells as a virtualized div grid (not <td>), so target
   everything inside the component — headers AND virtual body cells. */
#lib-table, #lib-table * { font-size: 12px !important; }
#lib-table .cell-wrap { padding: 2px 5px !important; line-height: 1.25 !important; }
"""

with gr.Blocks(title="Transcript Library") as demo:
    visible_state = gr.State([])   # dict-list currently shown in the table
    current_json = gr.State("")

    stats_md = gr.Markdown("Loading…")
    with gr.Row():
        url_in = gr.Textbox(label="Add a video", placeholder="YouTube URL or ID", scale=4)
        fetch_btn = gr.Button("Fetch", scale=1, variant="primary")
    fetch_msg = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=5):
            search_in = gr.Textbox(label="Search", placeholder="Search transcripts…")
            mode = gr.Radio(["Keyword", "Semantic"], value="Keyword", label="Mode")
            search_note = gr.Markdown("")
            table = gr.Dataframe(headers=LIB_HEADERS, interactive=False, wrap=True,
                                 elem_id="lib-table", column_widths=LIB_WIDTHS,
                                 max_height=460)
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
                chat_btn = gr.Button("Send", variant="primary")
                chat_out = gr.Markdown("", sanitize_html=False)
            with gr.Tab("Settings"):
                s = library.load_settings()
                # Embedding source is independent of the chat API: "local" uses
                # fastembed; anything else (legacy openai/ollama/api) means "api".
                _emb_source = "local" if s["embedding_provider"] == "local" else "api"
                gr.Markdown("### Embeddings")
                prov = gr.Radio(
                    [("Local — fastembed, no API", "local"), ("Use the API below", "api")],
                    value=_emb_source, label="Embedding source")
                emodel = gr.Dropdown(
                    choices=sorted({library.LOCAL_EMBED_MODEL, s["embedding_model"]}),
                    value=s["embedding_model"], label="Embedding model",
                    allow_custom_value=True)
                gr.Markdown("### Chat &amp; API connection\n"
                            "Used for chat. Also used for embeddings only if "
                            "\"Use the API\" is selected above.")
                api_type = gr.Radio([("OpenAI", "openai"), ("Ollama", "ollama")],
                                     value=s.get("api_type", "openai"),
                                     label="API type")
                base = gr.Textbox(s["api_base_url"], label="API base URL",
                                  placeholder="https://api.openai.com/v1")
                key = gr.Textbox(s["api_key"], label="API key", type="password")
                discover_btn = gr.Button("Discover models", variant="primary")
                discover_msg = gr.Markdown("")
                cmodel = gr.Dropdown(
                    choices=[s["chat_model"]] if s["chat_model"] else [],
                    value=s["chat_model"] or None, label="Chat model",
                    allow_custom_value=True)
                num_ctx = gr.Number(
                    value=s.get("num_ctx") or 0, precision=0, minimum=0,
                    label="num_ctx (Ollama context window)",
                    info="Ollama only; 0 = server default. When > 0, sent to "
                         "Ollama's native /api/chat endpoint (which honours it, "
                         "unlike the /v1 OpenAI-compatible endpoint).")
                think_cb = gr.Checkbox(
                    value=s.get("think", False), label="Stream reasoning (think)",
                    info="Ollama sends think:true; OpenAI shows reasoning only if "
                         "the server emits it. Only affects models that support it.")
                save_btn = gr.Button("Save settings", variant="primary")
                save_msg = gr.Markdown("")

    # Two separate load handlers: one pure-JS to define window.ytSeek on the
    # client (executes; a <script> in gr.HTML would not), one for the data.
    # Kept separate so the no-return js can't clobber the data fn's outputs.
    demo.load(None, None, None, js=SEEK_JS)
    demo.load(load_library, outputs=[visible_state, table, stats_md])
    fetch_btn.click(do_fetch, [url_in], [fetch_msg, table, stats_md, visible_state])
    search_in.submit(do_search, [search_in, mode], [table, search_note, visible_state])
    mode.change(do_search, [search_in, mode], [table, search_note, visible_state])
    table.select(on_row_select, [visible_state, search_in],
                 [player_html, meta_md, md_view, current_json])
    chat_btn.click(do_chat, [chat_in, chat_scope, current_json], [chat_out])
    api_type.change(on_api_type_change, [api_type, base], [base])
    discover_btn.click(discover_models_ui, [base, key], [emodel, cmodel, discover_msg])
    save_btn.click(save_settings_ui,
                   [prov, emodel, api_type, base, key, cmodel, num_ctx, think_cb],
                   [save_msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, css=TABLE_CSS)
