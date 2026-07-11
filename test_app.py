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


def test_settings_non_dict_json_ignored(tmp_path, monkeypatch):
    """Verify that non-dict JSON in config.json is ignored, not raising TypeError."""
    monkeypatch.setattr(library, "config_dir", lambda: tmp_path)
    config_file = tmp_path / "config.json"

    # Write a non-dict JSON value (list)
    config_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    # Should return defaults without raising
    result = library.load_settings()
    assert result["embedding_provider"] == "local"

    # Test with other non-dict values
    config_file.write_text(json.dumps(42), encoding="utf-8")
    result = library.load_settings()
    assert result["embedding_provider"] == "local"

    config_file.write_text(json.dumps("string"), encoding="utf-8")
    result = library.load_settings()
    assert result["embedding_provider"] == "local"

    config_file.write_text(json.dumps(None), encoding="utf-8")
    result = library.load_settings()
    assert result["embedding_provider"] == "local"


# Task 4: Chunking + embedding cache + semantic search
import numpy as np


def test_chunk_snippets_windows():
    snips = [{"text": f"s{i}", "start": float(i * 10), "duration": 10.0} for i in range(10)]
    chunks = library.chunk_snippets(snips, window_seconds=45.0)
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["text"].startswith("s0")
    # first chunk spans starts 0,10,20,30,40 (<=45 from 0) -> next chunk starts at 50
    assert chunks[1]["start"] == 50.0


def test_semantic_search_ranks_by_cosine(tmp_path, monkeypatch):
    # Two snippets; fake embedder maps text -> fixed vectors so ranking is deterministic.
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa", [
        {"text": "cats and dogs", "start": 0.0, "duration": 40.0},
        {"text": "quantum physics", "start": 60.0, "duration": 40.0},
    ])
    vectors = {
        "cats and dogs": np.array([1.0, 0.0], dtype="float32"),
        "quantum physics": np.array([0.0, 1.0], dtype="float32"),
        "kittens": np.array([0.9, 0.1], dtype="float32"),   # query
    }

    def fake_embed(texts, settings):
        return np.vstack([vectors[t] for t in texts])

    monkeypatch.setattr(library, "_EMBED_FN", fake_embed)
    monkeypatch.setattr(library, "load_settings",
                        lambda: {**library.DEFAULT_SETTINGS})
    hits = library.semantic_search(tmp_path, "kittens",
                                   library.DEFAULT_SETTINGS, top_k=2)
    assert hits[0]["text"].startswith("cats")   # closest to the query vector
    assert "score" in hits[0]


def test_default_embed_caches_local_model(monkeypatch):
    # Prove _default_embed builds the (slow, multi-second) ONNX model once per
    # model name, not once per call. Fake out fastembed entirely -- no real
    # model download.
    import sys
    import types

    build_count = {"n": 0}

    class FakeTextEmbedding:
        def __init__(self, model_name):
            build_count["n"] += 1
            self.model_name = model_name

        def embed(self, texts):
            return [np.zeros(3, dtype="float32") for _ in texts]

    fake_module = types.ModuleType("fastembed")
    fake_module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    monkeypatch.setattr(library, "_LOCAL_MODELS", {})

    settings = {**library.DEFAULT_SETTINGS, "embedding_provider": "local",
               "embedding_model": "same-model"}
    for _ in range(3):
        library._default_embed(["hello"], settings)

    assert build_count["n"] == 1


def test_cache_invalidates_on_model_change(tmp_path, monkeypatch):
    jp = _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                      [{"text": "hello", "start": 0.0, "duration": 5.0}])
    calls = {"n": 0}

    def fake_embed(texts, settings):
        calls["n"] += 1
        return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setattr(library, "_EMBED_FN", fake_embed)
    s1 = {**library.DEFAULT_SETTINGS, "embedding_model": "model-a"}
    s2 = {**library.DEFAULT_SETTINGS, "embedding_model": "model-b"}
    library.embed_video(str(jp), s1)
    library.embed_video(str(jp), s1)   # cached, no new embed call
    assert calls["n"] == 1
    library.embed_video(str(jp), s2)   # different model -> re-embed
    assert calls["n"] == 2


# Task 5: Chat over retrieved chunks
def test_build_chat_prompt_has_citation_instruction():
    hits = [{"title": "My Vid", "start": 65.0, "text": "hello world",
             "video_id": "x", "channel": "c", "json_path": "p"}]
    msgs = library.build_chat_prompt("what is said?", hits)
    assert msgs[0]["role"] == "system"
    assert "mm:ss" in msgs[0]["content"].lower() or "@" in msgs[0]["content"]
    assert "[My Vid @ 1:05]" in msgs[1]["content"]


def test_chat_calls_endpoint(tmp_path, monkeypatch):
    _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                 [{"text": "hello", "start": 0.0, "duration": 5.0}])
    monkeypatch.setattr(library, "semantic_search",
                        lambda *a, **k: [{"title": "Vid", "start": 0.0,
                                          "text": "hello", "video_id": "aaaaaaaaaaa",
                                          "channel": "Chan A", "json_path": "p"}])
    monkeypatch.setattr(library, "_CHAT_FN",
                        lambda messages, settings: "The answer [Vid @ 0:00]")
    s = {**library.DEFAULT_SETTINGS, "api_base_url": "http://x/v1", "chat_model": "m"}
    out = library.chat("q", tmp_path, s)
    assert "answer" in out


def test_chat_requires_config(tmp_path):
    try:
        library.chat("q", tmp_path, library.DEFAULT_SETTINGS)
        assert False, "expected error"
    except ValueError as e:
        assert "Settings" in str(e) or "endpoint" in str(e).lower()


def test_discover_models_local_only():
    out = library.discover_models({"embedding_provider": "local", "api_base_url": ""})
    assert out["embedding"] == [library.LOCAL_EMBED_MODEL]
    assert out["chat"] == []


def test_discover_models_api_always_includes_local(monkeypatch):
    monkeypatch.setattr(library, "_MODELS_FN",
                        lambda s: ["gpt-4o", "text-embedding-3-small"])
    out = library.discover_models({"embedding_provider": "openai",
                                   "api_base_url": "http://x/v1", "api_key": "k"})
    assert library.LOCAL_EMBED_MODEL in out["embedding"]          # always offered
    assert "text-embedding-3-small" in out["embedding"]
    assert out["chat"] == ["gpt-4o", "text-embedding-3-small"]


def test_local_embeddings_never_call_api(monkeypatch):
    # Regression: embedding_provider="local" must use fastembed, never the API,
    # even when an api_base_url is configured (for chat).
    def boom(texts, settings):
        raise AssertionError("API must not be called for local embeddings")

    class FakeModel:
        def embed(self, texts):
            return [np.ones(3, dtype="float32") for _ in texts]

    monkeypatch.setattr(library, "_api_embed", boom)
    monkeypatch.setattr(library, "_get_local_model", lambda name: FakeModel())
    out = library._default_embed(
        ["hi"], {"embedding_provider": "local", "embedding_model": "x",
                 "api_base_url": "http://192.168.0.4:8080/v1"})
    assert out.shape == (1, 3)


def test_api_provider_uses_api(monkeypatch):
    monkeypatch.setattr(library, "_api_embed",
                        lambda texts, s: np.ones((len(texts), 2), dtype="float32"))
    out = library._default_embed(["a", "b"], {"embedding_provider": "api"})
    assert out.shape == (2, 2)


def test_default_model_ids_parses_and_sorts(monkeypatch):
    import requests

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"id": "b-model"}, {"id": "a-model"}, {"nope": 1}]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    ids = library._default_model_ids({"api_base_url": "http://x/v1", "api_key": ""})
    assert ids == ["a-model", "b-model"]   # sorted, id-less entry skipped


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
