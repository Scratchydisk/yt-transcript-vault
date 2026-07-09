#!/usr/bin/env python3
"""Fetch a YouTube video transcript and save it as raw JSON + AI-ready Markdown.

Usage:
    python transcribe.py "<youtube-url>" [--languages en de ...] [--output-dir transcripts]

Output layout:
    <output-dir>/<channel-slug>/<video-title>.json   # raw transcript data + metadata
    <output-dir>/<channel-slug>/<video-title>.md      # readable transcript for AI tools

The transcript itself comes from `youtube-transcript-api`. The video title and
channel name come from YouTube's public oEmbed endpoint (no API key required),
since youtube-transcript-api does not expose that metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

APP_DIR_NAME = "youtube-transcribe"


def default_data_dir() -> Path:
    """Where transcripts live: env override, else per-platform user data dir."""
    env = os.environ.get("YT_TRANSCRIBE_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def config_dir() -> Path:
    """Where config.json lives (same folder as data on Windows)."""
    if sys.platform == "win32":
        return default_data_dir()
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIR_NAME


OEMBED_URL = "https://www.youtube.com/oembed"


def extract_video_id(url: str) -> str:
    """Pull the 11-character video ID out of any common YouTube URL form."""
    url = url.strip()

    # Already a bare ID.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        else:
            # /shorts/<id>, /embed/<id>, /live/<id>, /v/<id>
            parts = [p for p in parsed.path.split("/") if p]
            candidate = parts[1] if len(parts) >= 2 else ""
    else:
        candidate = ""

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate

    raise ValueError(f"Could not extract a YouTube video ID from: {url!r}")


def fetch_metadata(video_url: str, video_id: str) -> dict:
    """Get title + channel name via oEmbed, with safe fallbacks."""
    try:
        resp = requests.get(
            OEMBED_URL,
            params={"url": video_url, "format": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title") or video_id,
            "channel": data.get("author_name") or "unknown-channel",
            "channel_url": data.get("author_url", ""),
        }
    except (requests.RequestException, ValueError) as exc:
        print(f"  ! Could not fetch video metadata ({exc}); using fallbacks.",
              file=sys.stderr)
        return {"title": video_id, "channel": "unknown-channel", "channel_url": ""}


def fetch_publish_date(video_url: str) -> str:
    """Read the publish date (YYYY-MM-DD) from the watch page; '' if unavailable.

    oEmbed does not expose the publish date, so we scrape the watch page, where
    YouTube embeds it as `"publishDate":"..."` (and a `datePublished` meta tag).
    """
    try:
        resp = requests.get(
            video_url,
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
            timeout=15,
        )
        resp.raise_for_status()
        match = re.search(r'"publishDate":"([^"]+)"', resp.text) or re.search(
            r'itemprop="datePublished"[^>]*content="([^"]+)"', resp.text
        )
        if match:
            return match.group(1)[:10]  # ISO datetime -> date only
    except requests.RequestException as exc:
        print(f"  ! Could not fetch publish date ({exc}).", file=sys.stderr)
    return ""


def slugify(value: str, max_length: int | None = None) -> str:
    """Slugify text for use in paths.

    'Dark Finance' -> 'dark-finance', "You Can't Run It" -> 'you-cant-run-it'.
    Used for both the channel folder name and the video file name.
    """
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower()
    value = re.sub(r"['’]", "", value)            # drop apostrophes: can't -> cant
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if max_length is not None and len(value) > max_length:
        value = value[:max_length].rstrip("-")
    return value or "unknown"


def paragraphize(text: str, sentences_per_paragraph: int = 4) -> str:
    """Group a flat transcript into readable paragraphs at sentence boundaries.

    Transcripts arrive as one long run of text; this splits on sentence-ending
    punctuation and joins every few sentences into a paragraph separated by a
    blank line, so the Markdown isn't a single enormous line.
    """
    sentences = re.findall(r"[^.!?]+(?:[.!?]+|$)", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text
    paragraphs = [
        " ".join(sentences[i:i + sentences_per_paragraph])
        for i in range(0, len(sentences), sentences_per_paragraph)
    ]
    return "\n\n".join(paragraphs)


def build_markdown(meta: dict, video_id: str, video_url: str, transcript) -> str:
    """Produce a clean, metadata-headed Markdown doc ready for AI consumption."""
    full_text = " ".join(snippet.text.strip() for snippet in transcript if snippet.text.strip())
    full_text = re.sub(r"\s+", " ", full_text).strip()
    full_text = paragraphize(full_text)

    header = [
        f"# {meta['title']}",
        "",
        f"- **Channel:** {meta['channel']}",
        f"- **Published:** {meta.get('published') or 'unknown'}",
        f"- **Video URL:** {video_url}",
        f"- **Video ID:** {video_id}",
        f"- **Language:** {transcript.language} ({transcript.language_code})",
        f"- **Auto-generated:** {'yes' if transcript.is_generated else 'no'}",
        "",
        "## Transcript",
        "",
        full_text,
        "",
    ]
    return "\n".join(header)


def transcribe(video_url: str, languages: list[str], output_dir: Path) -> Path:
    video_id = extract_video_id(video_url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"Video ID: {video_id}")

    meta = fetch_metadata(canonical_url, video_id)
    meta["published"] = fetch_publish_date(canonical_url)
    print(f"Title:   {meta['title']}")
    print(f"Channel: {meta['channel']}")
    print(f"Published: {meta['published'] or 'unknown'}")

    print("Fetching transcript (this may take a moment)...")
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=languages)
    except NoTranscriptFound:
        # Requested languages unavailable — fall back to whatever exists.
        transcript_list = ytt_api.list(video_id)
        transcript = next(iter(transcript_list)).fetch()
        print(f"  ! No transcript in {languages}; using '{transcript.language_code}'.",
              file=sys.stderr)

    channel_dir = output_dir / slugify(meta["channel"])
    channel_dir.mkdir(parents=True, exist_ok=True)
    base = channel_dir / slugify(meta["title"], max_length=120)

    raw_payload = {
        "title": meta["title"],
        "channel": meta["channel"],
        "channel_url": meta["channel_url"],
        "published": meta.get("published", ""),
        "video_id": video_id,
        "video_url": canonical_url,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "snippets": transcript.to_raw_data(),
    }

    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(meta, video_id, canonical_url, transcript), encoding="utf-8")

    print(f"Saved:   {json_path}")
    print(f"Saved:   {md_path}")
    print("Done.")
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a YouTube transcript as JSON + Markdown.")
    parser.add_argument("url", help="YouTube video URL (or bare 11-char video ID)")
    parser.add_argument(
        "--languages", "-l", nargs="+", default=["en"],
        help="Preferred transcript languages, in order (default: en)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Root output directory (default: per-platform user data dir)",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else default_data_dir()

    # Flush each line immediately so progress is visible during the slow network
    # fetch, even when output is piped or captured (otherwise it looks frozen).
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    try:
        transcribe(args.url, args.languages, output_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (TranscriptsDisabled, VideoUnavailable, CouldNotRetrieveTranscript) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
