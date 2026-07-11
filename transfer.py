"""Zip export/import of the transcript library — move a vault between PCs.

No Gradio import here — this module stays UI-agnostic and unit-testable,
exactly like library.py. All paths are relative to the transcript data root.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

# An importable transcript entry: 'channel-slug/name.json' — exactly one slash,
# non-empty parts. Excludes top-level files (config.json) and nested junk.
_ENTRY_RE = __import__("re").compile(r"^([^/]+)/([^/]+)\.json$")


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


def resolve_selection(entries: list[dict], selected_channels: list[str],
                      selected_videos: list[str]) -> list[str]:
    """Union of ticked videos and all new videos under ticked channels (deduped)."""
    wanted: list[str] = []
    seen: set[str] = set()
    channel_set = set(selected_channels)
    for e in entries:
        if e["channel_slug"] in channel_set and not e["duplicate"]:
            arc = e["json_arcname"]
            if arc not in seen:
                seen.add(arc); wanted.append(arc)
    for arc in selected_videos:
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
