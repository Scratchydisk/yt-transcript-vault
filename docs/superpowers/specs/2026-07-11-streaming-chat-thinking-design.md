# Streaming chat responses (with reasoning/thinking)

**Date:** 2026-07-11
**Status:** Approved — ready for implementation plan

## Goal

Stream chat responses token-by-token in the Gradio UI instead of waiting for the
full reply, and surface model "thinking"/reasoning tokens in a collapsible block
above the answer. Works for both chat backends: the OpenAI-compatible
`/v1/chat/completions` endpoint and Ollama's native `/api/chat`.

## Decisions (locked)

- **Backends:** both OpenAI-compatible (SSE) and Ollama native (NDJSON).
- **Thinking UI:** collapsible `<details>` block above the answer; open while the
  answer is still empty, collapsed once answer tokens start arriving.
- **Thinking control:** a persisted Settings toggle, **off by default**.
- **OpenAI reasoning caveat:** there is no portable request param to *ask* an
  OpenAI-compatible server for reasoning. The toggle sends a real `think:true` to
  Ollama; for the OpenAI path the toggle only gates *displaying* reasoning deltas
  (`reasoning_content` / `reasoning`) if the server happens to emit them.

## Architecture

### Layering

- **`library.py`** owns network + text; stays UI-agnostic.
  - New generator `chat_stream(query, root, settings, only_json=None, top_k=8)`.
    Does retrieval + prompt build (identical to `chat()`), delegates to a backend
    streamer, and **yields cumulative `(thinking, answer)` string tuples**. The
    `answer` element is passed through `link_citations()` on each yield so
    citations become links as soon as they complete.
- **`app.py`** owns presentation.
  - `do_chat` consumes `chat_stream` and renders Markdown each yield.

### Backend streamers (`library.py`)

Parallel to the existing blocking `_default_chat` / `_ollama_chat`. Each is a
generator yielding `(kind, delta)` where `kind ∈ {"thinking", "content"}`.

- `_default_chat_stream(messages, settings)`
  - POST `{api_base_url}/chat/completions` with `stream: true`.
  - `requests.post(..., stream=True)`; `resp.raise_for_status()` before iterating.
  - Iterate `resp.iter_lines()`; for each `data: {...}` line (skip `data: [DONE]`
    and blank lines), parse JSON, read `choices[0].delta`:
    - `delta.get("content")` → `("content", text)`
    - `delta.get("reasoning_content") or delta.get("reasoning")` → `("thinking", text)`
  - Auth header as today (`Bearer {api_key}`).
- `_ollama_chat_stream(messages, settings)`
  - Derive native base from stored `/v1` base (reuse the existing `/v1`-stripping
    logic), POST `{host}/api/chat` with `stream: true`.
  - Payload includes `options.num_ctx` (when `> 0`, unchanged behaviour) and
    `think: true` **only when** `settings["think"]` is set.
  - Auth header only if `api_key` is set (matches current `_ollama_chat`).
  - Iterate `resp.iter_lines()`; each line is a JSON object with a `message`:
    - `message.get("content")` → `("content", text)`
    - `message.get("thinking")` → `("thinking", text)`

### `chat_stream` accumulation

```
thinking, answer = "", ""
for kind, delta in stream:
    if kind == "thinking":
        thinking += delta
    else:
        answer += delta
    yield thinking, link_citations(answer, hits)
```

`link_citations` on partial text is safe: incomplete `[title @ mm:ss]` citations
simply don't match yet and render as plain text until complete.

### Test seam

Add `_CHAT_STREAM_FN = None` mirroring the existing `_CHAT_FN`. `chat_stream`
uses `_CHAT_STREAM_FN or _default_chat_stream`/`_ollama_chat_stream` selection by
`api_type`. (Selection by `api_type` stays inside `chat_stream`; the override, if
set, replaces the backend streamer entirely for tests.)

### Presentation (`app.py`)

`do_chat` becomes:

```
yield "_Thinking…_"
try:
    only = current_json if scope == "This video" and current_json else None
    last = "_Thinking…_"
    for thinking, answer in library.chat_stream(query, ROOT, library.load_settings(), only_json=only):
        last = render_chat(thinking, answer)
        yield last
except Exception as exc:
    yield f"Error: {exc}"
```

`render_chat(thinking, answer)` helper:
- If `thinking` is non-empty, prepend
  `<details{ ' open' if not answer else ''}><summary>Thinking…</summary>\n\n{thinking}\n\n</details>`.
- Followed by the `answer` markdown.
- If both empty, return `"_Thinking…_"`.

**Rendering risk to verify during build:** Gradio's `gr.Markdown` may sanitise
`<details>`/`<summary>`. Verify; if stripped, set `sanitize_html=False` on the
`chat_out` Markdown component. Fallback if that's unworkable: render reasoning as
a `>` blockquote section headed `**Thinking…**`.

### Settings

- `DEFAULT_SETTINGS["think"] = False`.
- Settings tab: `think_cb = gr.Checkbox(value=s.get("think", False),
  label="Stream reasoning (think)", info="Ollama sends think:true; OpenAI shows
  reasoning only if the server emits it. Only affects models that support it.")`.
- `save_settings_ui` gains a `think` parameter, saved into the settings dict;
  `save_btn.click` inputs list gains `think_cb`.

## Backward compatibility

- Blocking `chat()`, `_default_chat`, `_ollama_chat` are left unchanged (CLI use
  and existing `_CHAT_FN` test injection).
- `think:true` is only ever sent when the toggle is on, so existing Ollama
  configs with non-reasoning models are unaffected by default.
- `num_ctx` behaviour in the Ollama path is preserved exactly.

## Testing

- Unit: inject `_CHAT_STREAM_FN` to yield a scripted sequence of
  `("thinking", …)` / `("content", …)` events; assert `chat_stream` yields
  cumulative tuples and that citations in the answer get linked.
- Manual verification (no automated UI tests in this repo):
  - Ollama reasoning model with toggle on → thinking block streams, then answer.
  - Ollama non-reasoning model with toggle off → answer streams, no thinking.
  - OpenAI-compatible endpoint → answer streams; reasoning appears only if emitted.
  - Confirm `<details>` renders in `gr.Markdown` (or apply the fallback).

## Out of scope

- Stop/cancel button for an in-flight stream.
- Persisting or re-rendering thinking after the turn completes (it stays in the
  final Markdown as a collapsed block; no separate history store).
- Streaming for the CLI (`transcribe.py` has no chat).
