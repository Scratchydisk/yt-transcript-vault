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
