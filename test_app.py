import os
from pathlib import Path
import importlib

import transcribe


def _reload(monkeypatch, **env):
    for k in ("YT_TRANSCRIBE_DIR", "APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_env_override_wins(monkeypatch, tmp_path):
    _reload(monkeypatch, YT_TRANSCRIBE_DIR=str(tmp_path / "custom"))
    assert transcribe.default_data_dir() == tmp_path / "custom"


def test_xdg_data_home(monkeypatch, tmp_path):
    _reload(monkeypatch, XDG_DATA_HOME=str(tmp_path / "xdg"))
    monkeypatch.setattr(transcribe.sys, "platform", "linux")
    assert transcribe.default_data_dir() == tmp_path / "xdg" / "youtube-transcribe"


def test_windows_appdata(monkeypatch, tmp_path):
    _reload(monkeypatch, APPDATA=str(tmp_path / "AppData"))
    monkeypatch.setattr(transcribe.sys, "platform", "win32")
    assert transcribe.default_data_dir() == tmp_path / "AppData" / "youtube-transcribe"


def test_config_dir_linux(monkeypatch, tmp_path):
    _reload(monkeypatch, XDG_CONFIG_HOME=str(tmp_path / "cfg"))
    monkeypatch.setattr(transcribe.sys, "platform", "linux")
    assert transcribe.config_dir() == tmp_path / "cfg" / "youtube-transcribe"


# Task 2: Library scan + keyword search
import json
import library


def _write_video(root, channel, title, vid, snippets, published="2026-01-01"):
    slug = channel.lower().replace(" ", "-")
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title, "channel": channel, "channel_url": "",
        "published": published, "video_id": vid,
        "video_url": f"https://www.youtube.com/watch?v={vid}",
        "language": "English", "language_code": "en", "is_generated": True,
        "snippets": snippets,
    }
    p = d / f"{title.lower().replace(' ', '-')}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_format_timestamp():
    assert library.format_timestamp(0) == "0:00"
    assert library.format_timestamp(65) == "1:05"
    assert library.format_timestamp(3725) == "1:02:05"


def test_scan_skips_corrupt(tmp_path, capsys):
    _write_video(tmp_path, "Chan A", "Hello World", "aaaaaaaaaaa",
                 [{"text": "hi", "start": 0.0, "duration": 1.0}])
    bad = tmp_path / "chan-a" / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    rows = library.scan_library(tmp_path)
    assert len(rows) == 1
    assert rows[0]["title"] == "Hello World"


def test_keyword_search_and_cap(tmp_path):
    snippets = [{"text": f"the word agent number {i}", "start": float(i), "duration": 1.0}
                for i in range(50)]
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa", snippets)
    hits, truncated = library.keyword_search(tmp_path, "AGENT", per_video_cap=20)
    assert len(hits) == 20
    assert truncated is True
    assert all("agent" in h["text"].lower() for h in hits)
    assert hits[0]["start"] == 0.0


def test_keyword_search_no_match(tmp_path):
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                 [{"text": "nothing here", "start": 0.0, "duration": 1.0}])
    hits, truncated = library.keyword_search(tmp_path, "zzz")
    assert hits == []
    assert truncated is False


def test_scan_library_sort_order(tmp_path):
    """Verify sort order: published descending, title ascending within same date."""
    snippets = [{"text": "test", "start": 0.0, "duration": 1.0}]

    # Create videos with the same published date but titles in reverse order
    # (Zebra before Apple alphabetically) to verify they get sorted ascending
    _write_video(tmp_path, "Chan A", "Zebra Video", "zzz1", snippets, published="2026-01-15")
    _write_video(tmp_path, "Chan A", "Apple Video", "aaa1", snippets, published="2026-01-15")

    # Create a video with a newer date to verify it sorts first
    _write_video(tmp_path, "Chan A", "Newest Video", "nnn1", snippets, published="2026-01-20")

    rows = library.scan_library(tmp_path)

    # Should have 3 videos
    assert len(rows) == 3

    # First should be the newest date
    assert rows[0]["title"] == "Newest Video"
    assert rows[0]["published"] == "2026-01-20"

    # Next two should be from 2026-01-15, sorted alphabetically
    assert rows[1]["title"] == "Apple Video"
    assert rows[1]["published"] == "2026-01-15"

    assert rows[2]["title"] == "Zebra Video"
    assert rows[2]["published"] == "2026-01-15"


# Task 3: Settings load/save
import stat


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "config_dir", lambda: tmp_path)
    s = library.load_settings()
    assert s["embedding_provider"] == "local"       # default
    s["api_key"] = "secret"
    library.save_settings(s)
    assert library.load_settings()["api_key"] == "secret"


def test_settings_file_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "config_dir", lambda: tmp_path)
    library.save_settings(library.DEFAULT_SETTINGS.copy())
    mode = stat.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600
