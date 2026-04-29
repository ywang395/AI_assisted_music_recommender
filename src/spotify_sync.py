"""
Syncs your Spotify data into songs.csv and history.jsonl.
Called automatically at terminal player startup when SPOTIFY_CLIENT_ID is set.

Sources pulled each sync:
  - Recently played   → skip/complete inferred from timestamp gaps vs track duration
  - Top tracks        → short/medium/long term (imported once per track, never duplicated)
  - Liked songs       → written as "repeat" events (strongest positive signal)

Dedup strategy:
  - Recently played   → deduped by Spotify's played_at timestamp
  - Top tracks/Liked  → deduped by spotify_id via data/.spotify_imported.json registry
"""

from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from src.models import InteractionEvent
from src.spotify_utils import CSV_COLUMNS, build_song_rows, estimate_features

_STEP_COUNTER: list[int] = [0]
_TOTAL_STEPS = 8


@contextmanager
def _step(label: str) -> Generator[None, None, None]:
    _STEP_COUNTER[0] += 1
    step_num = _STEP_COUNTER[0]
    t0 = time.perf_counter()
    print(f"  [step {step_num}/{_TOTAL_STEPS}] {label} ...", flush=True)
    yield
    elapsed = time.perf_counter() - t0
    print(f"  [step {step_num}/{_TOTAL_STEPS}] done ({elapsed*1000:.0f}ms)", flush=True)

IMPORT_REGISTRY_PATH = "data/.spotify_imported.json"
OAUTH_SCOPE = "user-read-recently-played user-top-read user-library-read"
TOP_TRACK_TERMS = ("short_term", "medium_term", "long_term")


# ── Registry (tracks which spotify IDs have already been imported) ────────────

def _load_registry() -> dict:
    if not Path(IMPORT_REGISTRY_PATH).exists():
        return {"top_tracks": [], "liked_songs": []}
    try:
        with open(IMPORT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {"top_tracks": [], "liked_songs": []}


def _save_registry(registry: dict) -> None:
    Path(IMPORT_REGISTRY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(IMPORT_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


# ── songs.csv helpers ─────────────────────────────────────────────────────────

def _load_songs_csv(path: str) -> Tuple[List[dict], bool]:
    if not Path(path).exists():
        return [], False
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), "spotify_id" in (reader.fieldnames or [])


def _known_track_keys(rows: List[dict]) -> set:
    return {(r["title"].strip().lower(), r["artist"].strip().lower()) for r in rows}


def _next_id(rows: List[dict]) -> int:
    return max((int(r.get("id", 0)) for r in rows), default=0) + 1


def _rewrite_csv(rows: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


# ── History helpers ───────────────────────────────────────────────────────────

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


def _make_event(
    event_type: str,
    session_id: str,
    song: dict,
    timestamp: str,
    elapsed_ratio: float = 1.0,
) -> InteractionEvent:
    duration = 30.0
    elapsed = round(duration * elapsed_ratio, 1)
    return InteractionEvent(
        event_type=event_type,
        session_id=session_id,
        song_id=int(song["id"]),
        song_title=song["title"],
        song_artist=song["artist"],
        song_genre=song["genre"],
        song_mood=song["mood"],
        song_energy=float(song["energy"]),
        song_valence=float(song["valence"]),
        song_tempo_bpm=float(song["tempo_bpm"]),
        song_score=0.0,
        elapsed_seconds=elapsed,
        total_duration=duration,
        elapsed_ratio=elapsed_ratio,
        repeat_count=0,
        timestamp=timestamp,
    )


# ── Skip inference from recently played ───────────────────────────────────────

def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _build_recent_events(
    items: list,
    song_lookup: dict,
    known_timestamps: set,
) -> List[InteractionEvent]:
    if not items:
        return []

    # Sort chronologically — Spotify returns newest first
    sorted_items = sorted(items, key=lambda x: x.get("played_at", ""))

    events = []
    for i, item in enumerate(sorted_items):
        played_at = item.get("played_at", "")
        if not played_at or played_at in known_timestamps:
            continue

        track = item["track"]
        key = (track["name"].strip().lower(), track["artists"][0]["name"].strip().lower())
        song = song_lookup.get(key)
        if song is None:
            continue

        duration_ms = track.get("duration_ms") or 210_000

        # Infer skip vs complete from gap to the next track
        if i + 1 < len(sorted_items):
            next_ts = sorted_items[i + 1].get("played_at", "")
            try:
                gap_ms = (_parse_ts(next_ts) - _parse_ts(played_at)).total_seconds() * 1000
                elapsed_ratio = round(min(1.0, max(0.0, gap_ms / duration_ms)), 2)
                event_type = "skip" if elapsed_ratio < 0.5 else "complete"
            except (ValueError, TypeError):
                elapsed_ratio, event_type = 1.0, "complete"
        else:
            elapsed_ratio, event_type = 1.0, "complete"

        events.append(_make_event(event_type, "spotify-recent", song, played_at, elapsed_ratio))

    return events


# ── Top tracks events ─────────────────────────────────────────────────────────

def _build_top_track_events(
    top_by_term: dict,
    song_lookup: dict,
    registry: dict,
) -> List[InteractionEvent]:
    imported = set(registry.get("top_tracks", []))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = []
    newly_imported = []

    # Process short_term first — if same track in multiple terms, short_term wins
    seen_in_batch: set = set()
    for term in TOP_TRACK_TERMS:
        for track in top_by_term.get(term, []):
            tid = track["id"]
            key = (track["name"].strip().lower(), track["artists"][0]["name"].strip().lower())
            if tid in imported or tid in seen_in_batch:
                continue
            song = song_lookup.get(key)
            if song is None:
                continue
            # Use a synthetic timestamp that encodes term so it's unique in history
            ts = f"spotify-top-{term}-{tid}"
            events.append(_make_event("complete", f"spotify-top-{term}", song, ts, 1.0))
            seen_in_batch.add(tid)
            newly_imported.append(tid)

    registry["top_tracks"] = list(imported | set(newly_imported))
    return events


# ── Liked songs events ────────────────────────────────────────────────────────

def _build_liked_events(
    liked_items: list,
    song_lookup: dict,
    registry: dict,
) -> List[InteractionEvent]:
    imported = set(registry.get("liked_songs", []))
    events = []
    newly_imported = []

    for item in liked_items:
        track = item.get("track") or item
        tid = track["id"]
        if tid in imported:
            continue
        key = (track["name"].strip().lower(), track["artists"][0]["name"].strip().lower())
        song = song_lookup.get(key)
        if song is None:
            continue
        added_at = item.get("added_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        # "repeat" = strongest positive signal (user explicitly hearted this track)
        events.append(_make_event("repeat", "spotify-liked", song, added_at, 1.0))
        newly_imported.append(tid)

    registry["liked_songs"] = list(imported | set(newly_imported))
    return events


# ── Collect all new tracks across sources ─────────────────────────────────────

def _collect_new_tracks(
    recent_items: list,
    top_by_term: dict,
    liked_items: list,
    known_keys: set,
) -> List[dict]:
    seen: set = set()
    new_tracks = []

    def _maybe_add(track: dict) -> None:
        key = (track["name"].strip().lower(), track["artists"][0]["name"].strip().lower())
        if key not in known_keys and key not in seen:
            new_tracks.append(track)
            seen.add(key)

    for item in recent_items:
        _maybe_add(item["track"])
    for term in TOP_TRACK_TERMS:
        for track in top_by_term.get(term, []):
            _maybe_add(track)
    for item in liked_items:
        _maybe_add(item.get("track") or item)

    return new_tracks


# ── Main entry point ──────────────────────────────────────────────────────────

def run_sync(
    songs_csv: str = "data/songs.csv",
    history_path: str = "data/history.jsonl",
    limit: int = 50,
) -> List[dict]:
    """
    Full Spotify sync: recently played + top tracks + liked songs.
    Emits numbered pipeline step logs for each observable stage.
    Returns updated songs list. Fails gracefully on any error.
    """
    _STEP_COUNTER[0] = 0  # reset step counter for each sync run

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

    # ── Step 1: Authenticate ──────────────────────────────────────────────────
    try:
        with _step("Spotify OAuth authentication"):
            sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=OAUTH_SCOPE,
                cache_path=".spotify_token_cache",
                open_browser=True,
            ))
    except Exception as exc:
        print(f"[sync] auth failed: {exc}")
        return _songs_as_dicts(songs_csv)

    # ── Step 2: Fetch recently played ─────────────────────────────────────────
    recent_items: list = []
    try:
        with _step("Fetch recently played tracks"):
            result = sp.current_user_recently_played(limit=min(limit, 50))
            recent_items = result.get("items", [])
            print(f"         → {len(recent_items)} tracks retrieved")
    except Exception as exc:
        print(f"[sync] recently-played failed: {exc}")

    # ── Step 3: Fetch top tracks (3 terms) ────────────────────────────────────
    top_by_term: dict = {}
    try:
        with _step("Fetch top tracks (short / medium / long term)"):
            for term in TOP_TRACK_TERMS:
                result = sp.current_user_top_tracks(limit=50, time_range=term)
                top_by_term[term] = result.get("items", [])
            total_top = sum(len(v) for v in top_by_term.values())
            print(f"         → {total_top} tracks ({'/'.join(str(len(top_by_term[t])) for t in TOP_TRACK_TERMS)} short/med/long)")
    except Exception as exc:
        print(f"[sync] top-tracks failed: {exc}")

    # ── Step 4: Fetch liked songs ─────────────────────────────────────────────
    liked_items: list = []
    try:
        with _step("Fetch liked songs"):
            result = sp.current_user_saved_tracks(limit=50)
            liked_items = result.get("items", [])
            print(f"         → {len(liked_items)} tracks retrieved")
    except Exception as exc:
        print(f"[sync] liked-songs failed: {exc}")

    # ── Step 5: Detect new tracks & estimate features ─────────────────────────
    existing_rows, has_spotify_id_col = _load_songs_csv(songs_csv)
    known_keys = _known_track_keys(existing_rows)
    new_tracks = _collect_new_tracks(recent_items, top_by_term, liked_items, known_keys)

    with _step(f"Detect new tracks + estimate audio features via OpenAI"):
        if new_tracks:
            print(f"         → {len(new_tracks)} new tracks found")
            if openai_key:
                openai_client = OpenAI(api_key=openai_key)
                estimates = estimate_features(new_tracks, openai_client, include_genre=True)
                new_rows = build_song_rows(new_tracks, estimates, _next_id(existing_rows))
                all_rows = existing_rows + new_rows
                _rewrite_csv(all_rows, songs_csv)
                print(f"         → songs.csv updated: {len(all_rows)} total (+{len(new_rows)})")
                existing_rows = all_rows
            else:
                print("         → WARNING: OPENAI_API_KEY not set — skipping feature estimation")
        else:
            print(f"         → 0 new tracks (library up to date at {len(existing_rows)} songs)")
            if not has_spotify_id_col:
                _rewrite_csv(existing_rows, songs_csv)

    # ── Step 6: Skip inference from recently played ───────────────────────────
    song_lookup = {
        (r["title"].strip().lower(), r["artist"].strip().lower()): r
        for r in existing_rows
    }
    known_timestamps = _load_history_timestamps(history_path)
    registry = _load_registry()

    with _step("Skip inference from recently-played timestamps"):
        recent_events = _build_recent_events(recent_items, song_lookup, known_timestamps)
        skips = sum(1 for e in recent_events if e.event_type == "skip")
        completes = sum(1 for e in recent_events if e.event_type == "complete")
        print(f"         → {len(recent_events)} events: {completes} complete, {skips} skip")

    # ── Step 7: Build top-track + liked-song events ───────────────────────────
    with _step("Build top-track + liked-song history events"):
        top_events = _build_top_track_events(top_by_term, song_lookup, registry)
        liked_events = _build_liked_events(liked_items, song_lookup, registry)
        print(f"         → {len(top_events)} top-track events, {len(liked_events)} liked events")

    # ── Step 8: Append all events to history.jsonl ────────────────────────────
    all_new_events = recent_events + top_events + liked_events
    with _step("Append new events to history.jsonl"):
        if all_new_events:
            from src.llm_reeval import append_events_to_history
            append_events_to_history(all_new_events, history_path)
            print(f"         → {len(all_new_events)} events written")
        else:
            print("         → no new events to append")

    _save_registry(registry)
    return _songs_as_dicts(songs_csv)


def _songs_as_dicts(songs_csv: str) -> List[dict]:
    from src.recommender import load_songs
    try:
        return load_songs(songs_csv)
    except Exception:
        return []
