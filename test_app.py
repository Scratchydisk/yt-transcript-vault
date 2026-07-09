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
