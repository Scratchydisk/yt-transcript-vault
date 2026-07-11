# Streaming Chat with Reasoning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream chat responses token-by-token in the Gradio UI, surfacing model reasoning ("thinking") in a collapsible block above the answer, for both the OpenAI-compatible and Ollama-native chat backends.

**Architecture:** `library.py` gains a `chat_stream()` generator that runs the existing retrieval + prompt build, delegates to a per-backend streaming generator, and yields cumulative `(thinking, answer)` string tuples with citations linked on each yield. `app.py`'s `do_chat` consumes that generator and renders Markdown each step; a pure `render_chat()` helper wraps thinking in a `<details>` block. A persisted `think` setting (off by default) sends `think:true` to Ollama and gates reasoning display.

**Tech Stack:** Python 3.14 (venv), `requests` (streaming via `iter_lines`), Gradio 6.20, pytest.

## Global Constraints

- British English in all user-facing copy and comments.
- Target **youtube-transcript-api 1.x**; venv Python is 3.14.4.
- Test command: `venv/bin/python -m pytest test_app.py -q` (pytest is installed in the venv; it is a dev-only dependency — do **not** add it to `requirements.txt`).
- Tests follow the existing `test_app.py` conventions: pytest fixtures (`tmp_path`, `monkeypatch`), grouped under `# Task N` comments, no network access (fake `requests`/inject `_CHAT_*_FN`).
- Preserve existing blocking `chat()`, `_default_chat`, `_ollama_chat` unchanged (CLI + existing tests depend on them).
- `think:true` and reasoning display only ever engage when the `think` setting is on; default behaviour is unchanged.

---

### Task 1: `chat_stream` orchestration + `think` setting + test seam

Adds the streaming entry point that later tasks plug backends into. Fully testable now via an injected fake streamer — no network.

**Files:**
- Modify: `library.py` — `DEFAULT_SETTINGS` (currently lines 91–98), add `_CHAT_STREAM_FN` near `_CHAT_FN` (line 288), add `chat_stream()` after `chat()` (after line 377).
- Test: `test_app.py` (append to the `# Task 5: Chat` group).

**Interfaces:**
- Consumes: existing `semantic_search`, `build_chat_prompt`, `link_citations`.
- Produces:
  - `DEFAULT_SETTINGS["think"] = False`
  - `_CHAT_STREAM_FN = None` — test seam; when set, replaces backend selection.
  - `chat_stream(query: str, root: Path, settings: dict, only_json: str | None = None, top_k: int = 8) -> Iterator[tuple[str, str]]` — yields cumulative `(thinking, answer)`, `answer` already run through `link_citations`. Selects backend by `settings["api_type"]` (`"ollama"` → `_ollama_chat_stream`, else `_default_chat_stream`) unless `_CHAT_STREAM_FN` is set. Backend streamers (Tasks 2–3) are generators yielding `(kind, delta)` with `kind ∈ {"thinking","content"}`.

- [ ] **Step 1: Write the failing tests**

Append to `test_app.py`:

```python
# Task 6: Streaming chat
def test_chat_stream_accumulates_thinking_and_links_citations(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "semantic_search",
                        lambda *a, **k: [{"title": "Vid", "start": 0.0,
                                          "text": "hello", "video_id": "aaaaaaaaaaa",
                                          "channel": "C", "json_path": "p"}])

    def fake_stream(messages, settings):
        yield "thinking", "let me "
        yield "thinking", "think"
        yield "content", "See [Vid @ 0:00]"

    monkeypatch.setattr(library, "_CHAT_STREAM_FN", fake_stream)
    s = {**library.DEFAULT_SETTINGS, "api_base_url": "http://x/v1", "chat_model": "m"}
    frames = list(library.chat_stream("q", tmp_path, s))
    assert frames[0] == ("let me ", "")                      # thinking streams first
    assert frames[-1][0] == "let me think"                   # thinking accumulates
    assert "youtube.com/watch?v=aaaaaaaaaaa" in frames[-1][1]  # citation linked


def test_chat_stream_requires_config(tmp_path):
    try:
        list(library.chat_stream("q", tmp_path, library.DEFAULT_SETTINGS))
        assert False, "expected error"
    except ValueError as e:
        assert "Settings" in str(e) or "endpoint" in str(e).lower()


def test_default_settings_has_think_off():
    assert library.DEFAULT_SETTINGS["think"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest test_app.py -k "chat_stream or think_off" -q`
Expected: FAIL — `AttributeError: module 'library' has no attribute 'chat_stream'` / `_CHAT_STREAM_FN`, and `KeyError: 'think'`.

- [ ] **Step 3: Add the `think` default**

In `library.py`, `DEFAULT_SETTINGS`, add the key after `chat_model` (keep the existing `num_ctx` line):

```python
    "chat_model": "",
    "think": False,                             # request/show model reasoning (Ollama: think=true)
    "num_ctx": 0,                               # Ollama context window; 0 = don't send (server default)
```

- [ ] **Step 4: Add the test seam**

In `library.py`, immediately after `_CHAT_FN = None  # tests inject; production uses _default_chat` (line 288):

```python
_CHAT_STREAM_FN = None  # tests inject; production uses _default/_ollama_chat_stream
```

- [ ] **Step 5: Add `chat_stream`**

In `library.py`, after the existing `chat()` function (after line 377):

```python
def chat_stream(query: str, root: Path, settings: dict, only_json: str | None = None,
                top_k: int = 8):
    """Streaming counterpart to chat(). Yields cumulative (thinking, answer)
    tuples; `answer` has citations linked on each yield so links appear as soon
    as a citation completes."""
    if not settings.get("api_base_url") or not settings.get("chat_model"):
        raise ValueError("Chat is not configured — set an API endpoint and "
                         "chat model in Settings.")
    hits = semantic_search(root, query, settings, top_k=top_k, only_json=only_json)
    messages = build_chat_prompt(query, hits)
    if _CHAT_STREAM_FN:
        streamer = _CHAT_STREAM_FN
    elif settings.get("api_type") == "ollama":
        streamer = _ollama_chat_stream
    else:
        streamer = _default_chat_stream
    thinking, answer = "", ""
    for kind, delta in streamer(messages, settings):
        if kind == "thinking":
            thinking += delta
        else:
            answer += delta
        yield thinking, link_citations(answer, hits)
```

Note: `_default_chat_stream` / `_ollama_chat_stream` are defined in Tasks 2–3. Task 1's tests only exercise the `_CHAT_STREAM_FN` branch, so they pass before those exist as long as the names are only referenced inside the function body (not at import time) — which they are.

- [ ] **Step 6: Run tests to verify they pass**

Run: `venv/bin/python -m pytest test_app.py -k "chat_stream or think_off" -q`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add library.py test_app.py
git commit -m "feat: chat_stream orchestrator + think setting"
```

---

### Task 2: `_default_chat_stream` — OpenAI-compatible SSE

**Files:**
- Modify: `library.py` — add after the blocking `_default_chat` (after line 324).
- Test: `test_app.py`.

**Interfaces:**
- Produces: `_default_chat_stream(messages: list[dict], settings: dict) -> Iterator[tuple[str, str]]`. POSTs to `{api_base_url}/chat/completions` with `stream: True`; yields `("thinking", d)` for `delta.reasoning_content`/`delta.reasoning` and `("content", d)` for `delta.content`; stops at `data: [DONE]`.

- [ ] **Step 1: Write the failing test**

Append to `test_app.py` (Task 6 group):

```python
def test_default_chat_stream_parses_sse(monkeypatch):
    import sys, types

    class FakeResp:
        def raise_for_status(self): pass
        def iter_lines(self, decode_unicode=False):
            return iter([
                'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
                '',
                'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                'data: [DONE]',
                'data: {"choices":[{"delta":{"content":"IGNORED"}}]}',
            ])

    captured = {}

    def fake_post(url, **kw):
        captured.update(url=url, json=kw.get("json"), stream=kw.get("stream"),
                        headers=kw.get("headers"))
        return FakeResp()

    fake = types.ModuleType("requests"); fake.post = fake_post
    monkeypatch.setitem(sys.modules, "requests", fake)

    s = {"api_base_url": "http://x/v1", "api_key": "k", "chat_model": "m"}
    events = list(library._default_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("thinking", "hmm"), ("content", "Hi")]
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["json"]["stream"] is True
    assert captured["stream"] is True
    assert captured["headers"] == {"Authorization": "Bearer k"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest test_app.py -k default_chat_stream -q`
Expected: FAIL — `AttributeError: module 'library' has no attribute '_default_chat_stream'`.

- [ ] **Step 3: Implement `_default_chat_stream`**

In `library.py`, after `_default_chat` (after line 324). `json` is already imported at module top.

```python
def _default_chat_stream(messages: list[dict], settings: dict):
    import requests
    payload = {"model": settings["chat_model"], "messages": messages, "stream": True}
    resp = requests.post(
        settings["api_base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {settings['api_key']}"},
        json=payload,
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"]
        except (ValueError, KeyError, IndexError):
            continue
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            yield "thinking", reasoning
        content = delta.get("content")
        if content:
            yield "content", content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest test_app.py -k default_chat_stream -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add library.py test_app.py
git commit -m "feat: OpenAI-compatible SSE streaming"
```

---

### Task 3: `_ollama_chat_stream` — native NDJSON

**Files:**
- Modify: `library.py` — add after the blocking `_ollama_chat` (after line 355, before `_CITE_RE`).
- Test: `test_app.py`.

**Interfaces:**
- Produces: `_ollama_chat_stream(messages: list[dict], settings: dict) -> Iterator[tuple[str, str]]`. Derives native host from the stored `/v1` base, POSTs `{host}/api/chat` with `stream: True`, `options.num_ctx` when `num_ctx > 0`, and `think: True` when `settings["think"]`; yields `("thinking", d)` for `message.thinking` and `("content", d)` for `message.content`.

- [ ] **Step 1: Write the failing tests**

Append to `test_app.py` (Task 6 group):

```python
def _fake_requests_capturing(monkeypatch, lines):
    import sys, types

    class FakeResp:
        def raise_for_status(self): pass
        def iter_lines(self, decode_unicode=False):
            return iter(lines)

    captured = {}

    def fake_post(url, **kw):
        captured.update(url=url, json=kw.get("json"), headers=kw.get("headers"))
        return FakeResp()

    fake = types.ModuleType("requests"); fake.post = fake_post
    monkeypatch.setitem(sys.modules, "requests", fake)
    return captured


def test_ollama_chat_stream_flags_and_parsing(monkeypatch):
    captured = _fake_requests_capturing(monkeypatch, [
        '{"message":{"thinking":"reason"}}',
        '{"message":{"content":"Answer"}}',
        '{"message":{},"done":true}',
    ])
    s = {"api_base_url": "http://localhost:11434/v1", "api_key": "", "chat_model": "m",
         "num_ctx": 4096, "think": True}
    events = list(library._ollama_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("thinking", "reason"), ("content", "Answer")]
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["stream"] is True
    assert captured["json"]["think"] is True
    assert captured["json"]["options"] == {"num_ctx": 4096}
    assert captured["headers"] == {}                       # no key → no auth header


def test_ollama_chat_stream_omits_think_and_options_by_default(monkeypatch):
    captured = _fake_requests_capturing(monkeypatch, ['{"message":{"content":"Hi"}}'])
    s = {"api_base_url": "http://localhost:11434/v1", "api_key": "sk", "chat_model": "m",
         "num_ctx": 0, "think": False}
    events = list(library._ollama_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("content", "Hi")]
    assert "think" not in captured["json"]
    assert "options" not in captured["json"]
    assert captured["headers"] == {"Authorization": "Bearer sk"}  # key present → header
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest test_app.py -k ollama_chat_stream -q`
Expected: FAIL — `AttributeError: module 'library' has no attribute '_ollama_chat_stream'`.

- [ ] **Step 3: Implement `_ollama_chat_stream`**

In `library.py`, after `_ollama_chat` (after line 355):

```python
def _ollama_chat_stream(messages: list[dict], settings: dict):
    # Native /api/chat streams NDJSON and honours options/think, unlike the /v1
    # OpenAI-compatible endpoint. See _ollama_chat for the /v1-base derivation.
    import requests
    base = settings["api_base_url"].rstrip("/")
    if base.endswith("/v1"):
        base = base[:-len("/v1")]
    payload = {"model": settings["chat_model"], "messages": messages, "stream": True}
    num_ctx = settings.get("num_ctx") or 0
    if num_ctx:
        payload["options"] = {"num_ctx": int(num_ctx)}
    if settings.get("think"):
        payload["think"] = True
    headers = {"Authorization": f"Bearer {settings['api_key']}"} \
        if settings.get("api_key") else {}
    resp = requests.post(base.rstrip("/") + "/api/chat", headers=headers,
                         json=payload, stream=True, timeout=120)
    resp.raise_for_status()
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        try:
            msg = json.loads(raw).get("message") or {}
        except ValueError:
            continue
        if msg.get("thinking"):
            yield "thinking", msg["thinking"]
        if msg.get("content"):
            yield "content", msg["content"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest test_app.py -k ollama_chat_stream -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add library.py test_app.py
git commit -m "feat: Ollama native NDJSON streaming with think"
```

---

### Task 4: Wire the UI — `render_chat`, streaming `do_chat`, Settings toggle

**Files:**
- Modify: `app.py` — add `render_chat` helper; rewrite `do_chat` (lines 138–146); extend `save_settings_ui` (lines 149–155); add `think_cb` checkbox in Settings (after `num_ctx`, ~line 249); set `sanitize_html=False` on `chat_out` (line 216); extend `save_btn.click` inputs (line 266).
- Test: `test_app.py`.

**Interfaces:**
- Consumes: `library.chat_stream`, `library.save_settings`, `s.get("think", False)`.
- Produces: `render_chat(thinking: str, answer: str) -> str`.

- [ ] **Step 1: Write the failing test**

Append to `test_app.py` (new `# Task 6` group; `import app` at top of the block):

```python
import app  # noqa: E402  (builds the Blocks at import; no launch)


def test_render_chat_details_open_until_answer():
    assert app.render_chat("", "") == "_Thinking…_"
    only_thinking = app.render_chat("reasoning here", "")
    assert "<details open>" in only_thinking and "reasoning here" in only_thinking
    with_answer = app.render_chat("reasoning here", "the answer")
    assert "<details>" in with_answer and "<details open>" not in with_answer
    assert "the answer" in with_answer
    assert app.render_chat("", "just answer") == "just answer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest test_app.py -k render_chat -q`
Expected: FAIL — `AttributeError: module 'app' has no attribute 'render_chat'`.

- [ ] **Step 3: Add `render_chat` and rewrite `do_chat`**

In `app.py`, replace the current `do_chat` (lines 138–146) with the helper + streaming generator:

```python
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
```

- [ ] **Step 4: Extend `save_settings_ui`**

In `app.py`, replace `save_settings_ui` (lines 149–155):

```python
def save_settings_ui(provider, model, api_type, base, key, chat_model, num_ctx, think):
    library.save_settings({
        "embedding_provider": provider, "embedding_model": model,
        "api_type": api_type, "api_base_url": base, "api_key": key,
        "chat_model": chat_model, "num_ctx": int(num_ctx or 0),
        "think": bool(think),
    })
    return "Saved."
```

- [ ] **Step 5: Add the `think` checkbox, make `chat_out` render HTML, wire the save button**

In `app.py`:

Set `chat_out` to allow the `<details>` HTML (line 216):

```python
                chat_out = gr.Markdown("", sanitize_html=False)
```

Add the checkbox immediately after the `num_ctx` `gr.Number(...)` block (~line 249), before `save_btn`:

```python
                think_cb = gr.Checkbox(
                    value=s.get("think", False), label="Stream reasoning (think)",
                    info="Ollama sends think:true; OpenAI shows reasoning only if "
                         "the server emits it. Only affects models that support it.")
```

Extend the save-button inputs (line 266) to include `think_cb`:

```python
    save_btn.click(save_settings_ui,
                   [prov, emodel, api_type, base, key, cmodel, num_ctx, think_cb],
                   [save_msg])
```

- [ ] **Step 6: Run the test + import smoke check + full suite**

Run: `venv/bin/python -m pytest test_app.py -k render_chat -q`
Expected: PASS.

Run: `venv/bin/python -c "import app; print('import OK')"`
Expected: prints `import OK` (confirms `sanitize_html=False`, the checkbox, and the reworked `save_btn.click` all construct without error).

Run: `venv/bin/python -m pytest test_app.py -q`
Expected: PASS (all tests — 24 prior + 8 new = 32).

- [ ] **Step 7: Commit**

```bash
git add app.py test_app.py
git commit -m "feat: stream chat responses with collapsible thinking"
```

---

### Task 5: Update docs

**Files:**
- Modify: `CLAUDE.md` — Chat and Settings descriptions.

- [ ] **Step 1: Update the Chat + Settings bullets**

In `CLAUDE.md`, under `library.py` → Chat, add streaming; under `app.py` → Settings, note the `think` toggle. Replace the `- **Chat**:` line:

```markdown
  - **Chat**: retrieves top-K chunks, calls the OpenAI-compatible endpoint (SSE)
    or Ollama's native `/api/chat` (NDJSON) and **streams** the reply; cites
    sources as `[title @ mm:ss]`. Reasoning ("thinking") is streamed into a
    collapsible block above the answer when the `think` setting is on.
```

And extend the `- **Settings**` line under `app.py` to mention the toggle:

```markdown
  - **Settings**: embedding provider (local/API), OpenAI-compatible base URL + API key (masked) + embedding/chat model names; API type (OpenAI/Ollama); num_ctx (Ollama); "Stream reasoning (think)" toggle; saved to `config.json`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note streaming chat and think toggle"
```

---

## Manual verification (no automated UI/network tests in this repo)

After Task 4, with a running Ollama (`./app.sh`):
1. Settings → API type **Ollama**, base `http://localhost:11434/v1`, pick a reasoning model, tick **Stream reasoning (think)**, Save.
2. Chat → confirm answer streams token-by-token and a collapsible **Thinking…** block appears (open while reasoning, collapsed once the answer starts), citations become clickable links.
3. Untick think + non-reasoning model → answer streams, no thinking block.
4. Switch to an OpenAI-compatible endpoint → answer streams; reasoning appears only if that server emits `reasoning_content`.
5. Confirm the `<details>` block actually renders (not shown as raw tags). If it renders as raw text, the `sanitize_html=False` on `chat_out` is missing or the Gradio build strips it — fall back to a `>` blockquote headed `**Thinking…**` in `render_chat`.

## Self-review notes

- **Spec coverage:** both backends (Tasks 2–3), collapsible thinking open-until-answer (Task 4 `render_chat`), persisted off-by-default toggle (Tasks 1 + 4), OpenAI display-only reasoning (Task 2 reads deltas, never sends a request param), Ollama `think:true` only when on (Task 3), citation linking during stream (Task 1), backward-compat blocking fns untouched (all tasks additive). Covered.
- **Type consistency:** backend streamers yield `(kind, delta)`; `chat_stream` yields `(thinking, answer)`; `render_chat(thinking, answer) -> str`. Consistent across tasks.
- **No placeholders:** all steps contain full code and exact commands.
