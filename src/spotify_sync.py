"""
Syncs your Spotify listening history into songs.csv and history.jsonl.
Called automatically at terminal player startup when SPOTIFY_CLIENT_ID is set.

Flow:
  1. Fetch recently played tracks from your Spotify account (OAuth, cached token)
  2. New tracks → OpenAI estimates audio features → appended to songs.csv
  3. All plays → written as "complete" events to history.jsonl (deduped by timestamp)
  4. Returns updated songs list ready for the recommender
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

from src.models import InteractionEvent
from src.spotify_utils import CSV_COLUMNS, build_song_rows, estimate_features, release_decade


def _load_songs_csv(path: str) -> Tuple[List[dict], bool]:
    """Return (rows, has_spotify_id_column)."""
    if not Path(path).exists():
        return [], False
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, "spotify_id" in fieldnames


def _known_track_keys(rows: List[dict]) -> set:
    return {(r["title"].strip().lower(), r["artist"].strip().lower()) for r in rows}


def _next_id(rows: List[dict]) -> int:
    if not rows:
        return 1
    return max(int(r.get("id", 0)) for r in rows) + 1


def _rewrite_csv(rows: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def _load_history_timestamps(history_path: str) -> set:
    if not Path(history_path).exists():
        return set()
    timestamps = set()
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ts = json.loads(line).get("timestamp", "")
                if ts:
                    timestamps.add(ts)
            except json.JSONDecodeError:
                continue
    return timestamps


def _make_complete_event(played_at: str, song: dict) -> InteractionEvent:
    duration = 30.0
    return InteractionEvent(
        event_type="complete",
        session_id="spotify-import",
        song_id=int(song["id"]),
        song_title=song["title"],
        song_artist=song["artist"],
        song_genre=song["genre"],
        song_mood=song["mood"],
        song_energy=float(song["energy"]),
        song_valence=float(song["valence"]),
        song_tempo_bpm=float(song["tempo_bpm"]),
        song_score=0.0,
        elapsed_seconds=duration,
        total_duration=duration,
        elapsed_ratio=1.0,
        repeat_count=0,
        timestamp=played_at,
    )


def run_sync(
    songs_csv: str = "data/songs.csv",
    history_path: str = "data/history.jsonl",
    limit: int = 50,
) -> List[dict]:
    """
    Sync Spotify recently played into songs.csv and history.jsonl.
    Returns the updated songs list. Fails gracefully on any error.
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not client_id or client_id == "your_client_id_here" or not client_secret:
        return _songs_as_dicts(songs_csv)

    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        from openai import OpenAI
    except ImportError as exc:
        print(f"[sync] skipped: missing package ({exc})")
        return _songs_as_dicts(songs_csv)

    # ── 1. Authenticate and fetch recently played ─────────────────────────────
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-read-recently-played",
            cache_path=".spotify_token_cache",
            open_browser=True,
        ))
        result = sp.current_user_recently_played(limit=min(limit, 50))
    except Exception as exc:
        print(f"[sync] Spotify auth/fetch failed: {exc}")
        return _songs_as_dicts(songs_csv)

    items = result.get("items", [])
    if not items:
        print("[sync] No recently played tracks found.")
        return _songs_as_dicts(songs_csv)

    print(f"[sync] Fetched {len(items)} recently played tracks from Spotify.")

    # ── 2. Find tracks not yet in songs.csv ──────────────────────────────────
    existing_rows, has_spotify_id_col = _load_songs_csv(songs_csv)
    known_keys = _known_track_keys(existing_rows)

    seen_in_batch: set = set()
    new_tracks = []
    for item in items:
        track = item["track"]
        key = (track["name"].strip().lower(), track["artists"][0]["name"].strip().lower())
        if key not in known_keys and key not in seen_in_batch:
            new_tracks.append(track)
            seen_in_batch.add(key)

    # ── 3. Estimate features for new tracks and rewrite songs.csv ────────────
    if new_tracks:
        print(f"[sync] {len(new_tracks)} new tracks — estimating audio features...")
        if not openai_key:
            print("[sync] WARNING: OPENAI_API_KEY not set, skipping feature estimation.")
            new_rows = []
        else:
            openai_client = OpenAI(api_key=openai_key)
            estimates = estimate_features(new_tracks, openai_client, include_genre=True)
            new_rows = build_song_rows(new_tracks, estimates, _next_id(existing_rows))

        if new_rows:
            all_rows = existing_rows + new_rows
            _rewrite_csv(all_rows, songs_csv)
            print(f"[sync] Added {len(new_rows)} new tracks to {songs_csv} (total: {len(all_rows)})")
            existing_rows = all_rows
    elif not has_spotify_id_col:
        # First sync on original CSV: rewrite to add spotify_id column to header
        _rewrite_csv(existing_rows, songs_csv)

    # ── 4. Write new play events to history.jsonl ─────────────────────────────
    song_lookup = {
        (r["title"].strip().lower(), r["artist"].strip().lower()): r
        for r in existing_rows
    }
    known_timestamps = _load_history_timestamps(history_path)
    new_events: list = []

    for item in items:
        played_at = item.get("played_at", "")
        if not played_at or played_at in known_timestamps:
            continue
        track = item["track"]
        key = (track["name"].strip().lower(), track["artists"][0]["name"].strip().lower())
        song = song_lookup.get(key)
        if song is None:
            continue
        new_events.append(_make_complete_event(played_at, song))

    if new_events:
        from src.llm_reeval import append_events_to_history
        append_events_to_history(new_events, history_path)
        print(f"[sync] Wrote {len(new_events)} new play events to {history_path}")
    else:
        print("[sync] No new play events to add.")

    return _songs_as_dicts(songs_csv)


def _songs_as_dicts(songs_csv: str) -> List[dict]:
    from src.recommender import load_songs
    try:
        return load_songs(songs_csv)
    except Exception:
        return []
