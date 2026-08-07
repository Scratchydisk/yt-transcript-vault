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
        status_code = 200
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

    s = {"api_base_url": "http://x/v1", "api_key": "k", "chat_model": "m", "think": True}
    events = list(library._default_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("thinking", "hmm"), ("content", "Hi")]
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["json"]["stream"] is True
    assert captured["stream"] is True
    assert captured["headers"] == {"Authorization": "Bearer k"}


def test_default_chat_stream_suppresses_reasoning_when_think_off(monkeypatch):
    import sys, types

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def iter_lines(self, decode_unicode=False):
            return iter([
                'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
                'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                'data: [DONE]',
            ])

    def fake_post(url, **kw):
        return FakeResp()

    fake = types.ModuleType("requests"); fake.post = fake_post
    monkeypatch.setitem(sys.modules, "requests", fake)

    s = {"api_base_url": "http://x/v1", "api_key": "k", "chat_model": "m"}  # no think
    events = list(library._default_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("content", "Hi")]


def _fake_requests_capturing(monkeypatch, lines):
    import sys, types

    class FakeResp:
        status_code = 200
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
    monkeypatch.setattr(library, "_ollama_supports_thinking", lambda s: True)
    s = {"api_base_url": "http://localhost:11434/v1", "api_key": "", "chat_model": "m",
         "num_ctx": 4096, "think": True}
    events = list(library._ollama_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("thinking", "reason"), ("content", "Answer")]
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["stream"] is True
    assert captured["json"]["think"] is True
    assert captured["json"]["options"] == {"num_ctx": 4096}
    assert captured["headers"] == {}                       # no key → no auth header


def test_ollama_chat_stream_omits_think_when_model_unsupported(monkeypatch):
    # think is on, but the model lacks the 'thinking' capability → degrade
    # gracefully to a non-thinking request instead of letting Ollama 400.
    captured = _fake_requests_capturing(monkeypatch, ['{"message":{"content":"Hi"}}'])
    monkeypatch.setattr(library, "_ollama_supports_thinking", lambda s: False)
    s = {"api_base_url": "http://localhost:11434/v1", "api_key": "", "chat_model": "m",
         "think": True}
    events = list(library._ollama_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("content", "Hi")]
    assert "think" not in captured["json"]


def test_ollama_supports_thinking_reads_capabilities(monkeypatch):
    import sys, types
    monkeypatch.setattr(library, "_OLLAMA_CAPS_CACHE", {})

    class ShowResp:
        status_code = 200
        def __init__(self, caps): self._caps = caps
        def json(self): return {"capabilities": self._caps}

    caps_by_model = {"reason-m": ["thinking", "tools"], "plain-m": ["tools", "completion"]}

    def fake_post(url, **kw):
        assert url.endswith("/api/show")
        return ShowResp(caps_by_model[kw["json"]["model"]])

    fake = types.ModuleType("requests"); fake.post = fake_post
    fake.RequestException = Exception
    monkeypatch.setitem(sys.modules, "requests", fake)

    base = {"api_base_url": "http://h:11434/v1", "api_key": ""}
    assert library._ollama_supports_thinking({**base, "chat_model": "reason-m"}) is True
    assert library._ollama_supports_thinking({**base, "chat_model": "plain-m"}) is False


def test_raise_for_status_with_body_surfaces_server_message():
    class BadResp:
        status_code = 400
        url = "http://h/api/chat"
        text = '{"error":"\\"m\\" does not support thinking"}'
    try:
        library._raise_for_status_with_body(BadResp())
        assert False, "expected error"
    except RuntimeError as e:
        assert "400" in str(e) and "does not support thinking" in str(e)

    class OkResp:
        status_code = 200
    library._raise_for_status_with_body(OkResp())  # 2xx → no raise


def test_ollama_chat_stream_omits_think_and_options_by_default(monkeypatch):
    captured = _fake_requests_capturing(monkeypatch, ['{"message":{"content":"Hi"}}'])
    s = {"api_base_url": "http://localhost:11434/v1", "api_key": "sk", "chat_model": "m",
         "num_ctx": 0, "think": False}
    events = list(library._ollama_chat_stream([{"role": "user", "content": "hi"}], s))
    assert events == [("content", "Hi")]
    assert "think" not in captured["json"]
    assert "options" not in captured["json"]
    assert captured["headers"] == {"Authorization": "Bearer sk"}  # key present → header


import app  # noqa: E402  (builds the Blocks at import; no launch)


def test_render_chat_details_open_until_answer():
    assert app.render_chat("", "") == "_Thinking…_"
    only_thinking = app.render_chat("reasoning here", "")
    assert "<details open>" in only_thinking and "reasoning here" in only_thinking
    with_answer = app.render_chat("reasoning here", "the answer")
    assert "<details>" in with_answer and "<details open>" not in with_answer
    assert "the answer" in with_answer
    assert app.render_chat("", "just answer") == "just answer"


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


# ---------- notes & chat export ----------

def test_notes_roundtrip_and_keyword_search(tmp_path):
    jp = _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                      [{"text": "transcript line", "start": 0.0, "duration": 1.0}])
    library.save_notes(str(jp), "remember the xylophone bit\n\nsecond para")
    assert "xylophone" in library.load_notes(str(jp))

    hits, _ = library.keyword_search(tmp_path, "xylophone")
    assert len(hits) == 1
    assert hits[0]["text"].startswith("[note]")
    assert hits[0]["start"] == 0.0

    # Empty notes removes the file and the search hits with it.
    library.save_notes(str(jp), "   ")
    assert not library.notes_path(str(jp)).exists()
    assert library.keyword_search(tmp_path, "xylophone")[0] == []


def test_notes_included_in_embedding_and_cache_refresh(tmp_path, monkeypatch):
    jp = _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                      [{"text": "transcript line", "start": 0.0, "duration": 1.0}])

    def fake_embed(texts, settings):
        return np.array([[1.0, 0.0]] * len(texts), dtype="float32")

    monkeypatch.setattr(library, "_EMBED_FN", fake_embed)
    settings = library.DEFAULT_SETTINGS.copy()

    vectors, chunks = library.embed_video(str(jp), settings)
    assert len(chunks) == 1

    library.save_notes(str(jp), "my note about widgets")
    vectors, chunks = library.embed_video(str(jp), settings)
    assert len(chunks) == 2
    assert chunks[-1]["text"] == "[user note] my note about widgets"
    assert vectors.shape[0] == 2

    # Deleting notes leaves a fresh-looking cache with the wrong shape;
    # the shape guard must re-embed rather than return a mismatched cache.
    library.save_notes(str(jp), "")
    vectors, chunks = library.embed_video(str(jp), settings)
    assert len(chunks) == 1
    assert vectors.shape[0] == 1


def test_export_chat_appends(tmp_path):
    jp = _write_video(tmp_path, "Chan A", "Vid", "aaaaaaaaaaa",
                      [{"text": "hi", "start": 0.0, "duration": 1.0}])
    p = library.export_chat(str(jp), "## first question\n\nanswer one")
    assert p == Path(str(jp)).with_suffix(".chat.md")
    library.export_chat(str(jp), "## second question\n\nanswer two")
    text = p.read_text(encoding="utf-8")
    assert text.index("first question") < text.index("---") < text.index("second question")
