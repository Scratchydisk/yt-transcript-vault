# Export / Import Transfer + Table-Refresh Fix — Design

**Date:** 2026-07-11
**Status:** Approved, ready for planning

## Problem

Three requests:

1. **Refresh the table when a video has been transcribed.** Reported as
   intermittent ("fails sometimes").
2. **Export the user's transcriptions as a zip** so they can be moved to
   another PC.
3. **Import selectively from another user's zip export** into your own storage.

## Root cause of (1)

The Fetch *button* is wired (`fetch_btn.click(do_fetch, …)`, `app.py:285`) and
`do_fetch` already reloads the whole library after a transcribe. But the **"Add a
video" textbox has no `submit` handler** — pressing **Enter** in it does nothing.
Because the Search box directly below *does* submit on Enter, pressing Enter in
the URL box is the natural habit, so the table appears not to refresh "sometimes"
(= whenever the user hits Enter instead of clicking Fetch).

Likely secondary contributor: a freshly-fetched video with an unknown publish
date sorts to the **bottom** of the list — `scan_library` sorts by `published`
descending and empty dates sort last (`library.py:52-53`) — so even a successful
refresh can look like nothing happened because the new row isn't where the user
looks.

## Design

### 1. Table-refresh fix

- Wire `url_in.submit(do_fetch, [url_in], [fetch_msg, table, stats_md,
  visible_state])` — identical inputs/outputs to `fetch_btn.click`, so Enter and
  click behave the same.
- `do_fetch` echoes the fetched **title** in its status message
  ("Fetched: *Title*") so a successful refresh is unmistakable, and reports
  clearly when the video was already present.

Out of scope: changing the library sort order. (Noted as a secondary factor; the
submit handler is the actual bug.)

### 2. New module `transfer.py`

Pure logic, **no Gradio import** — mirrors `library.py` so it stays
unit-testable. All paths are relative to the transcript data root
(`default_data_dir()`), passed in by the caller.

```
export_zip(root: Path, include_config: bool, dest: Path) -> Path
```
- Walk `root` for every `<channel>/<name>.json` and its `<name>.md` sibling; add
  to the zip with arcnames `channel-slug/name.json` (paths preserved).
- Add `config.json` at the zip root **only** when `include_config` is true.
  Excluded by default so shared zips never leak API keys.
- Returns the written zip path.

```
inspect_zip(zip_path: Path, root: Path) -> list[dict]
```
- Enumerate `*/*.json` entries in the zip. For each, derive
  `{channel_slug, title, json_arcname, md_arcname, duplicate}` where `title`
  comes from the JSON's `title` field (fallback: filename stem) and
  `duplicate = (root / channel_slug / f"{name}.json").exists()`.
- Ignore entries that don't fit the `channel/name.json` shape and any
  `config.json`.

```
import_selected(zip_path: Path, root: Path, selected_arcnames: list[str])
    -> {"imported": int, "skipped": int, "errors": list[str]}
```
- For each selected json arcname, extract it and its `.md` sibling into
  `root/<channel_slug>/`.
- **Skip existing** (conflict policy): if the destination json already exists,
  count it as skipped and do not overwrite.
- **Zip-slip protection:** reject any arcname that is absolute or contains `..`
  after normalisation; such entries go to `errors`, never written. These zips
  come from other users and are untrusted.

### 3. UI — new "Transfer" tab

Added to the right-panel tab group (beside Viewer / Markdown / Chat / Settings)
in `app.py`.

**Export.** An "Export library" button reveals a small inline panel (Gradio
6.20 has no core modal component, so a reveal-`gr.Group` is the robust equivalent
of a popup). The panel contains:
- a checkbox **"Include config.json (contains API keys)"** — default **off**, and
- a "Build download" button that calls `export_zip(...)` to a temp file and
  populates a `gr.DownloadButton` / `gr.File` for the user to download.

So the include-config choice is presented at download time, as requested.

**Import (both levels).** A `gr.File` upload restricted to `.zip`. On upload,
`inspect_zip` drives:
- a **channels** `gr.CheckboxGroup` (tick a whole channel), choices = distinct
  channel slugs that have at least one *new* video;
- a **videos** `gr.CheckboxGroup`, choices labelled `channel — title`, value =
  json arcname, containing only *new* (non-duplicate) videos;
- a status `gr.Markdown`: *"N new, M already in your library (will be skipped)."*

"Import selected" imports the **union** of (ticked videos) ∪ (all new videos
under ticked channels), calls `import_selected(...)`, then refreshes the table,
stats, and `visible_state` via the existing `load_library()` — exactly like a
fetch. It reports "Imported N, skipped M" (+ any errors).

Duplicates are excluded from the selectable choices (they'd be skipped anyway)
but counted in the status line.

**Chosen over:** dynamically-rendered per-channel accordions (`gr.render` with a
nested `CheckboxGroup` per channel). The two-checkbox-group approach gives the
same channel-level + video-level selection with far less dynamic-component
wiring and is more robust in Gradio.

### 4. Tests (`test_app.py`)

Unit tests against `transfer.py` (pure logic, no Gradio needed):
- `export_zip` includes every `<channel>/<name>.{json,md}`, **excludes**
  `config.json` by default and **includes** it when `include_config=True`.
- `inspect_zip` correctly flags duplicates vs. new entries.
- `import_selected` extracts new files, **skips** existing ones (no overwrite),
  and **rejects** zip-slip arcnames (`../…`, absolute) into `errors` without
  writing outside `root`.

## Non-goals

- No overwrite/merge conflict UI — skip-existing only.
- No selective *export* (whole library only).
- No change to library sort order.
- No import of `config.json` (never written on import, even if present in a zip).
