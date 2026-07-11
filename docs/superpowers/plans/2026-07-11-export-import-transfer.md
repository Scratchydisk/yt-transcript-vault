# Export / Import Transfer + Table-Refresh Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the intermittent table-refresh-after-fetch bug, and add the ability to export the whole transcript library as a zip and selectively import transcripts from another user's zip.

**Architecture:** A new UI-agnostic module `transfer.py` (mirroring `library.py`) holds all zip logic — export, inspect, selection resolution, and import — so it is fully unit-testable without Gradio. `app.py` gains a "Transfer" tab that wires these functions to Gradio components, and a one-line submit handler that fixes the refresh bug. Import refreshes the table by reusing the existing `load_library()`.

**Tech Stack:** Python 3.14, Gradio 6.20, stdlib `zipfile`/`pathlib`/`tempfile`, pytest.

## Global Constraints

- British English in all user-facing copy and comments.
- Target `youtube-transcript-api` 1.x instance API — not touched here, but do not downgrade pins.
- `transfer.py` MUST NOT import `gradio` — keep it UI-agnostic and unit-testable, exactly like `library.py`.
- Transcript data root is `transcribe.default_data_dir()`; config is `library.settings_path()` (= `config_dir()/config.json`, mode `0600`). Never write `config.json` into the data root on import.
- Import zips are **untrusted** (they come from other people): every extracted path must be validated against zip-slip (`..` / absolute) before writing.
- Conflict policy is **skip-existing**: never overwrite a local transcript on import.
- Tests live in the single top-level `test_app.py` and run via `venv/bin/python -m pytest`. pytest is dev-only (not in `requirements.txt`).

---

### Task 1: Fix intermittent table-refresh (Enter in the URL box)

**Files:**
- Modify: `app.py` (function `do_fetch` ~line 122; handler wiring ~line 285)
- Test: `test_app.py`

**Interfaces:**
- Consumes: existing `do_fetch(url, progress)`, `load_library()`, `transcribe.transcribe`.
- Produces: nothing new for later tasks; `do_fetch` keeps its 4-tuple return shape `(message, table_update, stats, rows)`.

**Root cause:** `url_in` (the "Add a video" textbox) has no `.submit` handler — only `fetch_btn.click` is wired. Pressing Enter (natural, because the Search box below submits on Enter) does nothing, so the table "sometimes" doesn't refresh. Fix: wire `url_in.submit` to the same handler, and echo the fetched title so success is unmistakable.

- [ ] **Step 1: Write the failing test** — assert the URL textbox has a submit handler and `do_fetch` names the fetched title.

Add to `test_app.py` (near the other `app` tests at the end):

```python
def test_url_textbox_submits_on_enter():
    # Regression: pressing Enter in the "Add a video" box must trigger a fetch,
    # not silently do nothing. Verify a submit-style dependency targets do_fetch.
    triggers = [d for d in app.demo.fns.values()
                if getattr(d, "fn", None) is app.do_fetch]
    # do_fetch must be wired at least twice: the Fetch button click AND url_in.submit.
    assert len(triggers) >= 2


def test_do_fetch_reports_title(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "ROOT", tmp_path)

    def fake_transcribe(url, langs, root):
        _write_video(root, "Chan A", "My Great Video", "aaaaaaaaaaa",
                     [{"text": "hi", "start": 0.0, "duration": 1.0}])

    monkeypatch.setattr(app, "transcribe", fake_transcribe)
    msg, table, stats, rows = app.do_fetch("https://youtu.be/aaaaaaaaaaa")
    assert "My Great Video" in msg
    assert any(r["title"] == "My Great Video" for r in rows)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest test_app.py::test_url_textbox_submits_on_enter test_app.py::test_do_fetch_reports_title -v`
Expected: FAIL — `test_url_textbox_submits_on_enter` fails (only one trigger), `test_do_fetch_reports_title` fails ("Fetched." has no title).

- [ ] **Step 3: Make `do_fetch` report the fetched title**

In `app.py`, replace the success branch of `do_fetch` so it names the newest matching video. Current body:

```python
    try:
        progress(0.3, desc="Fetching transcript…")
        transcribe(url, ["en"], ROOT)
        rows, table, stats = load_library()
        return "Fetched.", table, stats, rows
```

becomes:

```python
    try:
        progress(0.3, desc="Fetching transcript…")
        transcribe(url, ["en"], ROOT)
        rows, table, stats = load_library()
        vid = extract_video_id(url) if url.strip() else ""
        match = next((r for r in rows if r["video_id"] == vid), None)
        msg = f"Fetched: {match['title']}" if match else "Fetched."
        return msg, table, stats, rows
```

And add `extract_video_id` to the existing import at the top of `app.py`:

```python
from transcribe import default_data_dir, extract_video_id, transcribe
```

- [ ] **Step 4: Wire `url_in.submit`**

In `app.py`, immediately after the existing line (~285):

```python
    fetch_btn.click(do_fetch, [url_in], [fetch_msg, table, stats_md, visible_state])
```

add:

```python
    # Enter in the URL box must fetch too — the Search box below submits on Enter,
    # so users habitually press Enter here; without this the table "won't refresh".
    url_in.submit(do_fetch, [url_in], [fetch_msg, table, stats_md, visible_state])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest test_app.py::test_url_textbox_submits_on_enter test_app.py::test_do_fetch_reports_title -v`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
git add app.py test_app.py
git commit -m "fix: fetch on Enter in URL box and report fetched title"
```

---

### Task 2: `transfer.export_zip` — bundle the library

**Files:**
- Create: `transfer.py`
- Test: `test_app.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `export_zip(root: Path, dest: Path, include_config: bool = False, config_path: Path | None = None) -> Path` — writes a zip of every `<channel>/<name>.json` + `<name>.md` under `root` (arcnames `channel/name.ext`), adds `config.json` at the zip root only when `include_config` is true and `config_path` exists, returns `dest`.

- [ ] **Step 1: Write the failing test**

Add to `test_app.py` (new section header comment, after the last test):

```python
# Transfer: export / import
import transfer
import zipfile


def test_export_zip_includes_transcripts_excludes_config_by_default(tmp_path):
    root = tmp_path / "data"
    _write_video(root, "Dark Finance", "Money Talk", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    # Give it a sibling .md so we can assert both are bundled.
    (root / "dark-finance" / "money-talk.md").write_text("# md", encoding="utf-8")
    cfg = tmp_path / "config.json"
    cfg.write_text('{"api_key": "secret"}', encoding="utf-8")

    dest = tmp_path / "out.zip"
    result = transfer.export_zip(root, dest, include_config=False, config_path=cfg)

    assert result == dest
    names = set(zipfile.ZipFile(dest).namelist())
    assert "dark-finance/money-talk.json" in names
    assert "dark-finance/money-talk.md" in names
    assert "config.json" not in names            # excluded by default → no key leak


def test_export_zip_includes_config_when_asked(tmp_path):
    root = tmp_path / "data"
    _write_video(root, "Chan A", "Vid", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    cfg = tmp_path / "config.json"
    cfg.write_text('{"api_key": "secret"}', encoding="utf-8")

    dest = tmp_path / "out.zip"
    transfer.export_zip(root, dest, include_config=True, config_path=cfg)
    names = set(zipfile.ZipFile(dest).namelist())
    assert "config.json" in names
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest test_app.py -k export_zip -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transfer'`.

- [ ] **Step 3: Create `transfer.py` with `export_zip`**

```python
"""Zip export/import of the transcript library — move a vault between PCs.

No Gradio import here — this module stays UI-agnostic and unit-testable,
exactly like library.py. All paths are relative to the transcript data root.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


def export_zip(root: Path, dest: Path, include_config: bool = False,
               config_path: Path | None = None) -> Path:
    """Zip every <channel>/<name>.json + <name>.md under root into dest.

    Arcnames are 'channel-slug/name.ext' (the data root itself is stripped).
    config.json is added at the zip root ONLY when include_config is true and
    config_path exists — it holds API keys, so a shared export omits it by default.
    """
    root = Path(root)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.glob("*/*.json")) + sorted(root.glob("*/*.md")):
            zf.write(path, arcname=path.relative_to(root).as_posix())
        if include_config and config_path and Path(config_path).exists():
            zf.write(config_path, arcname="config.json")
    return Path(dest)
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest test_app.py -k export_zip -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add transfer.py test_app.py
git commit -m "feat: transfer.export_zip bundles the library to a zip"
```

---

### Task 3: `transfer.inspect_zip` — list entries and flag duplicates

**Files:**
- Modify: `transfer.py`
- Test: `test_app.py`

**Interfaces:**
- Consumes: `export_zip` (tests build a zip with it).
- Produces: `inspect_zip(zip_path: Path, root: Path) -> list[dict]`. Each dict:
  `{"channel_slug": str, "name": str, "title": str, "json_arcname": str, "md_arcname": str | None, "duplicate": bool}`.
  Only entries shaped `channel/name.json` (exactly one slash) are returned; `config.json` and malformed paths are ignored. `title` is the JSON's `title` field (fallback: `name`). `duplicate` is true when `root/channel/name.json` already exists locally.

- [ ] **Step 1: Write the failing test**

```python
def test_inspect_zip_lists_entries_and_flags_duplicates(tmp_path):
    # Build a source library and export it.
    src = tmp_path / "src"
    _write_video(src, "Chan A", "Alpha", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    _write_video(src, "Chan B", "Beta", "bbbbbbbbbbb",
                 [{"text": "yo", "start": 0.0, "duration": 1.0}])
    zip_path = tmp_path / "exp.zip"
    transfer.export_zip(src, zip_path)

    # Local root already has Chan A / Alpha → that entry is a duplicate.
    local = tmp_path / "local"
    _write_video(local, "Chan A", "Alpha", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])

    entries = transfer.inspect_zip(zip_path, local)
    by_arc = {e["json_arcname"]: e for e in entries}

    assert by_arc["chan-a/alpha.json"]["duplicate"] is True
    assert by_arc["chan-b/beta.json"]["duplicate"] is False
    assert by_arc["chan-b/beta.json"]["title"] == "Beta"
    assert by_arc["chan-b/beta.json"]["channel_slug"] == "chan-b"
    assert by_arc["chan-b/beta.json"]["md_arcname"] is None  # _write_video writes no .md
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest test_app.py -k inspect_zip -v`
Expected: FAIL — `AttributeError: module 'transfer' has no attribute 'inspect_zip'`.

- [ ] **Step 3: Add `inspect_zip` to `transfer.py`**

Add near the top (after imports):

```python
# An importable transcript entry: 'channel-slug/name.json' — exactly one slash,
# non-empty parts. Excludes top-level files (config.json) and nested junk.
_ENTRY_RE = __import__("re").compile(r"^([^/]+)/([^/]+)\.json$")
```

and append:

```python
def inspect_zip(zip_path: Path, root: Path) -> list[dict]:
    """List transcript entries in the zip, flagging which already exist under root."""
    root = Path(root)
    entries: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for name in sorted(names):
            m = _ENTRY_RE.match(name)
            if not m:
                continue
            channel_slug, stem = m.group(1), m.group(2)
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                title = data.get("title") or stem
            except (ValueError, OSError, KeyError):
                title = stem
            md_arcname = f"{channel_slug}/{stem}.md"
            entries.append({
                "channel_slug": channel_slug,
                "name": stem,
                "title": title,
                "json_arcname": name,
                "md_arcname": md_arcname if md_arcname in names else None,
                "duplicate": (root / channel_slug / f"{stem}.json").exists(),
            })
    return entries
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest test_app.py -k inspect_zip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transfer.py test_app.py
git commit -m "feat: transfer.inspect_zip lists entries and flags duplicates"
```

---

### Task 4: `transfer.import_selected` + `resolve_selection`

**Files:**
- Modify: `transfer.py`
- Test: `test_app.py`

**Interfaces:**
- Consumes: `export_zip`, `inspect_zip`.
- Produces:
  - `resolve_selection(entries: list[dict], selected_channels: list[str], selected_videos: list[str]) -> list[str]` — returns the deduped json arcnames to import: the union of `selected_videos` and every entry whose `channel_slug` is in `selected_channels`, **excluding duplicates**. Order is stable (first-seen).
  - `import_selected(zip_path: Path, root: Path, selected_arcnames: list[str]) -> dict` — extracts each selected json + its `.md` sibling into `root/<channel>/`, skipping any whose destination `.json` already exists. Rejects zip-slip / malformed arcnames into `errors`. Returns `{"imported": int, "skipped": int, "errors": list[str]}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_selection_unions_channels_and_videos_excluding_dupes():
    entries = [
        {"channel_slug": "chan-a", "json_arcname": "chan-a/alpha.json", "duplicate": False},
        {"channel_slug": "chan-a", "json_arcname": "chan-a/gamma.json", "duplicate": True},
        {"channel_slug": "chan-b", "json_arcname": "chan-b/beta.json", "duplicate": False},
    ]
    # Whole channel A (skips the duplicate gamma) + explicit beta video.
    out = transfer.resolve_selection(entries, ["chan-a"], ["chan-b/beta.json"])
    assert out == ["chan-a/alpha.json", "chan-b/beta.json"]


def test_import_selected_extracts_new_skips_existing(tmp_path):
    src = tmp_path / "src"
    _write_video(src, "Chan A", "Alpha", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    (src / "chan-a" / "alpha.md").write_text("# alpha", encoding="utf-8")
    _write_video(src, "Chan B", "Beta", "bbbbbbbbbbb",
                 [{"text": "yo", "start": 0.0, "duration": 1.0}])
    zip_path = tmp_path / "exp.zip"
    transfer.export_zip(src, zip_path)

    local = tmp_path / "local"
    # Pre-existing Alpha → must be skipped, not overwritten.
    _write_video(local, "Chan A", "Alpha", "aaaaaaaaaaa",
                 [{"text": "ORIGINAL", "start": 0.0, "duration": 1.0}])

    result = transfer.import_selected(
        zip_path, local, ["chan-a/alpha.json", "chan-b/beta.json"])

    assert result["imported"] == 1
    assert result["skipped"] == 1
    assert result["errors"] == []
    # Beta imported with its (absent) md → json present, no md written.
    assert (local / "chan-b" / "beta.json").exists()
    # Alpha untouched (still ORIGINAL) and its md NOT overwritten.
    assert "ORIGINAL" in (local / "chan-a" / "alpha.json").read_text(encoding="utf-8")


def test_import_selected_rejects_zip_slip(tmp_path):
    # Craft a malicious zip with a traversal path.
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.json", '{"title": "x"}')
        zf.writestr("chan-a/ok.json", '{"title": "ok"}')
    local = tmp_path / "local"
    local.mkdir()

    result = transfer.import_selected(
        zip_path, local, ["../escape.json", "chan-a/ok.json"])

    assert result["imported"] == 1                       # only the safe one
    assert any("escape" in e for e in result["errors"])  # traversal rejected
    assert not (tmp_path / "escape.json").exists()        # nothing escaped root
    assert (local / "chan-a" / "ok.json").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest test_app.py -k "resolve_selection or import_selected" -v`
Expected: FAIL — attributes not defined.

- [ ] **Step 3: Add `resolve_selection`, `import_selected`, and the safety helper**

Append to `transfer.py`:

```python
def resolve_selection(entries: list[dict], selected_channels: list[str],
                      selected_videos: list[str]) -> list[str]:
    """Union of ticked videos and all new videos under ticked channels (deduped)."""
    wanted: list[str] = []
    seen: set[str] = set()
    channel_set = set(selected_channels)
    for arc in selected_videos:
        if arc not in seen:
            seen.add(arc); wanted.append(arc)
    for e in entries:
        if e["channel_slug"] in channel_set and not e["duplicate"]:
            arc = e["json_arcname"]
            if arc not in seen:
                seen.add(arc); wanted.append(arc)
    return wanted


def _is_safe_arcname(name: str) -> bool:
    """channel/name.json with no traversal or absolute path — for untrusted zips."""
    if not _ENTRY_RE.match(name):
        return False
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts


def import_selected(zip_path: Path, root: Path,
                    selected_arcnames: list[str]) -> dict:
    """Extract selected json + .md siblings into root, skipping existing files.

    Untrusted input: any unsafe arcname (traversal/absolute/malformed) is rejected
    into errors and never written. Existing destinations are skipped, not overwritten.
    """
    root = Path(root)
    imported = skipped = 0
    errors: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for arc in selected_arcnames:
            if not _is_safe_arcname(arc):
                errors.append(f"Rejected unsafe path: {arc}")
                continue
            dest_json = root / arc
            if dest_json.exists():
                skipped += 1
                continue
            dest_json.parent.mkdir(parents=True, exist_ok=True)
            dest_json.write_bytes(zf.read(arc))
            md_arc = arc[:-len(".json")] + ".md"
            if md_arc in names:
                (root / md_arc).write_bytes(zf.read(md_arc))
            imported += 1
    return {"imported": imported, "skipped": skipped, "errors": errors}
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest test_app.py -k "resolve_selection or import_selected" -v`
Expected: PASS (all three).

- [ ] **Step 5: Run the whole transfer suite**

Run: `venv/bin/python -m pytest test_app.py -k "transfer or export_zip or inspect_zip or import_selected or resolve_selection" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add transfer.py test_app.py
git commit -m "feat: transfer.import_selected with skip-existing and zip-slip guard"
```

---

### Task 5: "Transfer" tab in the Gradio UI

**Files:**
- Modify: `app.py` (imports, new handler functions, new tab, handler wiring)
- Test: `test_app.py`

**Interfaces:**
- Consumes: `transfer.export_zip / inspect_zip / resolve_selection / import_selected`, existing `load_library()`, `library.settings_path()`, `ROOT`.
- Produces (module-level functions in `app.py`, so they're unit-testable):
  - `do_export(include_config: bool) -> dict` — builds the zip in a temp dir, returns `gr.update(value=<path>, visible=True)`.
  - `on_zip_upload(file) -> tuple` — returns updates for `(channels_cbg, videos_cbg, import_status, import_entries_state, import_zip_state)`.
  - `do_import(zip_path, entries, selected_channels, selected_videos) -> tuple` — returns `(import_msg, table_update, stats, rows)`.

- [ ] **Step 1: Write the failing tests** (unit-test the handler logic, not Gradio rendering)

```python
def test_do_export_builds_zip_without_config(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "ROOT", tmp_path / "data")
    _write_video(app.ROOT, "Chan A", "Vid", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    monkeypatch.setattr(app.library, "settings_path",
                        lambda: tmp_path / "nonexistent-config.json")

    upd = app.do_export(False)
    path = upd["value"]
    assert zipfile.ZipFile(path).namelist()  # non-empty zip produced
    assert "config.json" not in set(zipfile.ZipFile(path).namelist())


def test_on_zip_upload_populates_choices(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "ROOT", tmp_path / "local")
    src = tmp_path / "src"
    _write_video(src, "Chan A", "Alpha", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    zip_path = tmp_path / "exp.zip"
    transfer.export_zip(src, zip_path)

    class FakeFile:
        name = str(zip_path)

    chan_upd, vid_upd, status, entries, zpath = app.on_zip_upload(FakeFile())
    assert "chan-a" in chan_upd["choices"]
    assert any(v[1] == "chan-a/alpha.json" for v in vid_upd["choices"])  # (label, value)
    assert zpath == str(zip_path)
    assert "1 new" in status


def test_do_import_imports_and_refreshes(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "ROOT", tmp_path / "local")
    app.ROOT.mkdir(parents=True)
    src = tmp_path / "src"
    _write_video(src, "Chan A", "Alpha", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    zip_path = tmp_path / "exp.zip"
    transfer.export_zip(src, zip_path)
    entries = transfer.inspect_zip(zip_path, app.ROOT)

    msg, table, stats, rows = app.do_import(
        str(zip_path), entries, [], ["chan-a/alpha.json"])
    assert "Imported 1" in msg
    assert any(r["title"] == "Alpha" for r in rows)
    assert (app.ROOT / "chan-a" / "alpha.json").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest test_app.py -k "do_export or on_zip_upload or do_import" -v`
Expected: FAIL — `AttributeError: module 'app' has no attribute 'do_export'`.

- [ ] **Step 3: Add imports and handler functions to `app.py`**

At the top of `app.py`, add to the imports block:

```python
import tempfile
from datetime import date

import transfer
```

Add these handler functions (place them after `save_settings_ui`, before the `TABLE_CSS` block):

```python
def do_export(include_config: bool):
    # Build the zip in a temp dir; Gradio serves the returned path for download.
    out_dir = Path(tempfile.mkdtemp(prefix="yt-export-"))
    dest = out_dir / f"transcripts-export-{date.today().isoformat()}.zip"
    transfer.export_zip(ROOT, dest, include_config=bool(include_config),
                        config_path=library.settings_path())
    return gr.update(value=str(dest), visible=True)


def on_zip_upload(file):
    # file is a Gradio tempfile handle (has .name) or None when cleared.
    if not file:
        return (gr.update(choices=[], value=[]), gr.update(choices=[], value=[]),
                "", [], "")
    zip_path = file.name
    entries = transfer.inspect_zip(zip_path, ROOT)
    new = [e for e in entries if not e["duplicate"]]
    dupes = len(entries) - len(new)
    channels = sorted({e["channel_slug"] for e in new})
    video_choices = [(f"{e['channel_slug']} — {e['title']}", e["json_arcname"])
                     for e in new]
    status = f"{len(new)} new, {dupes} already in your library (will be skipped)."
    return (gr.update(choices=channels, value=[]),
            gr.update(choices=video_choices, value=[]),
            status, entries, zip_path)


def do_import(zip_path, entries, selected_channels, selected_videos):
    if not zip_path:
        rows, table, stats = load_library()
        return "Upload a zip first.", table, stats, rows
    arcnames = transfer.resolve_selection(
        entries or [], selected_channels or [], selected_videos or [])
    if not arcnames:
        rows, table, stats = load_library()
        return "Nothing selected.", table, stats, rows
    result = transfer.import_selected(zip_path, ROOT, arcnames)
    rows, table, stats = load_library()
    msg = f"Imported {result['imported']}, skipped {result['skipped']}."
    if result["errors"]:
        msg += f" {len(result['errors'])} rejected."
    return msg, table, stats, rows
```

- [ ] **Step 4: Add the "Transfer" tab**

In `app.py`, inside the right-hand `with gr.Column(scale=7):` tab group, after the `with gr.Tab("Settings"):` block (i.e. after `save_msg = gr.Markdown("")`, at the same indentation as the other `with gr.Tab(...)` lines), add:

```python
            with gr.Tab("Transfer"):
                gr.Markdown("### Export\nZip up your whole library to move it to "
                            "another PC.")
                export_btn = gr.Button("Export library…", variant="primary")
                with gr.Group(visible=False) as export_panel:
                    include_cfg = gr.Checkbox(
                        value=False,
                        label="Include config.json (contains your API keys)")
                    build_btn = gr.Button("Build download", variant="primary")
                    export_file = gr.File(label="Download", visible=False)
                gr.Markdown("### Import\nSelect another user's export and choose "
                            "what to bring in. Transcripts you already have are "
                            "skipped.")
                import_file = gr.File(label="Choose a zip export",
                                      file_types=[".zip"], type="filepath")
                import_status = gr.Markdown("")
                channels_cbg = gr.CheckboxGroup([], label="Whole channels")
                videos_cbg = gr.CheckboxGroup([], label="Individual videos")
                import_btn = gr.Button("Import selected", variant="primary")
                import_msg = gr.Markdown("")
                import_entries = gr.State([])
                import_zip = gr.State("")
```

Note: `gr.File(..., type="filepath")` makes the uploaded value a path string, so `on_zip_upload` must accept a string as well as an object. Adjust `on_zip_upload`'s first lines accordingly:

```python
def on_zip_upload(file):
    if not file:
        return (gr.update(choices=[], value=[]), gr.update(choices=[], value=[]),
                "", [], "")
    zip_path = file if isinstance(file, str) else file.name
```

- [ ] **Step 5: Wire the handlers**

In `app.py`, in the handler-wiring block at the bottom (after `save_btn.click(...)`), add:

```python
    export_btn.click(lambda: gr.update(visible=True), None, [export_panel])
    build_btn.click(do_export, [include_cfg], [export_file])
    import_file.change(on_zip_upload, [import_file],
                       [channels_cbg, videos_cbg, import_status,
                        import_entries, import_zip])
    import_btn.click(do_import,
                     [import_zip, import_entries, channels_cbg, videos_cbg],
                     [import_msg, table, stats_md, visible_state])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest test_app.py -k "do_export or on_zip_upload or do_import" -v`
Expected: PASS (all three).

- [ ] **Step 7: Run the full test suite (nothing regressed)**

Run: `venv/bin/python -m pytest test_app.py -v`
Expected: PASS (all tests, including the pre-existing ones).

- [ ] **Step 8: Smoke-test the UI builds**

Run: `venv/bin/python -c "import app; print('blocks built:', app.demo is not None)"`
Expected: prints `blocks built: True` with no exception (confirms the new tab/components construct).

- [ ] **Step 9: Commit**

```bash
git add app.py test_app.py
git commit -m "feat: Transfer tab for export/import of transcripts"
```

---

### Task 6: Docs

**Files:**
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `CLAUDE.md`**

In the **Architecture → Modules** section, add a bullet for the new module after the `library.py` bullet:

```markdown
- **`transfer.py`**: UI-agnostic zip export/import. `export_zip` bundles the
  whole library (config.json excluded unless asked); `inspect_zip` lists a zip's
  transcripts flagging duplicates; `import_selected` extracts chosen transcripts,
  skipping existing files and rejecting zip-slip paths from untrusted zips.
```

In the `app.py` layout bullet, note the new tab: change the tab list to
`Viewer / Markdown / Chat / Settings / Transfer` and add:

```markdown
  - **Transfer tab**: "Export library…" reveals an include-config choice then a
    download; import uploads a zip and offers whole-channel and per-video
    checkboxes (skip-existing) that refresh the table on import.
```

- [ ] **Step 2: Update `README.md`** — add a short "Moving your library to another PC" subsection describing Export (download a zip) and Import (upload a zip, tick channels/videos, existing skipped). Match the README's existing tone and heading style (open and read it first to match structure).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document transfer (export/import) feature"
```

---

## Self-Review

**Spec coverage:**
- Refresh-on-transcribe → Task 1 (root cause: missing `url_in.submit`). ✓
- Export as zip, config as a choice at download time → Task 2 (`export_zip`) + Task 5 (reveal-panel with `include_cfg` checkbox before `export_file`). ✓
- Selective import, both levels, skip-existing, untrusted-zip safety → Tasks 3–4 (`inspect_zip`, `resolve_selection`, `import_selected` with zip-slip guard) + Task 5 (channels + videos checkbox groups). ✓
- Tests for export/inspect/import → Tasks 2–5. ✓

**Placeholder scan:** none — every code step shows full code; README step says to read the file first to match tone (content-generation, not a code placeholder).

**Type consistency:** `export_zip(root, dest, include_config, config_path)`, `inspect_zip(zip_path, root) -> list[dict]` with keys `channel_slug/name/title/json_arcname/md_arcname/duplicate`, `resolve_selection(entries, selected_channels, selected_videos) -> list[str]`, `import_selected(zip_path, root, selected_arcnames) -> {"imported","skipped","errors"}` — used consistently across Tasks 3–5. `do_fetch` keeps its 4-tuple shape. `_ENTRY_RE` defined in Task 3 and reused by `_is_safe_arcname` in Task 4. ✓
```
