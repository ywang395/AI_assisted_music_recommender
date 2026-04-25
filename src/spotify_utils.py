"""
Shared utilities for Spotify integration.
Used by both spotify_fetch.py (genre-based population) and spotify_sync.py (history import).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

VALID_MOODS = [
    "happy", "joyful", "energetic", "intense", "angry", "motivated",
    "chill", "relaxed", "peaceful", "dreamy", "meditative", "focused",
    "sad", "melancholic", "moody", "romantic", "nostalgic",
]

# Maps internal genre names → Spotify search genre strings
GENRE_SEARCH_MAP = {
    "pop":        "pop",
    "lofi":       "lo-fi",
    "rock":       "rock",
    "ambient":    "ambient",
    "jazz":       "jazz",
    "indie pop":  "indie pop",
    "country":    "country",
    "electronic": "electronic",
    "folk":       "folk",
    "metal":      "metal",
    "latin":      "latin",
    "classical":  "classical",
    "blues":      "blues",
    "r&b":        "r&b",
    "hip-hop":    "hip hop",
    "synthwave":  "synthwave",
}

GENRE_VALUES = list(GENRE_SEARCH_MAP.keys())

CSV_COLUMNS = [
    "id", "title", "artist", "genre", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
    "popularity", "release_decade", "mood_tags",
    "live_energy", "lyrical_depth", "instrumentalness", "spotify_id",
]

_PROMPT_KEYS = (
    "title, artist, energy, tempo_bpm, valence, danceability, acousticness, "
    "instrumentalness, live_energy, lyrical_depth, mood, mood_tags"
)
_PROMPT_KEYS_WITH_GENRE = (
    "title, artist, genre, energy, tempo_bpm, valence, danceability, acousticness, "
    "instrumentalness, live_energy, lyrical_depth, mood, mood_tags"
)

_PROMPT_TEMPLATE = """\
You are a music analysis assistant. For each song below, estimate audio \
feature values and return ONLY a valid JSON array — no markdown, no prose.

Each element must have exactly these keys:
  {keys}

Rules:
- energy, valence, danceability, acousticness, instrumentalness,
  live_energy, lyrical_depth: float 0.0–1.0 (2 decimal places)
- tempo_bpm: integer 40–220{genre_rule}
- mood: one of {moods}
- mood_tags: 3 descriptive tags joined by "|" (e.g. "upbeat|bright|anthemic")

Songs:
{{songs}}
"""

_PROMPT_WITH_GENRE = _PROMPT_TEMPLATE.format(
    keys=_PROMPT_KEYS_WITH_GENRE,
    genre_rule="\n- genre: one of " + ", ".join(GENRE_VALUES),
    moods=", ".join(VALID_MOODS),
)

_PROMPT_NO_GENRE = _PROMPT_TEMPLATE.format(
    keys=_PROMPT_KEYS,
    genre_rule="",
    moods=", ".join(VALID_MOODS),
)


def release_decade(release_date: str) -> int:
    try:
        return (int(release_date[:4]) // 10) * 10
    except (ValueError, IndexError):
        return 2010


def estimate_features(tracks: list, openai_client, include_genre: bool = False) -> list:
    """
    Ask OpenAI to estimate audio features for a batch of Spotify tracks.

    include_genre=True  → used by spotify_sync (recently played, genre unknown)
    include_genre=False → used by spotify_fetch (genre already known from search)
    """
    if not tracks:
        return []

    songs_text = "\n".join(
        f'{i + 1}. "{t["name"]}" by {t["artists"][0]["name"]}'
        for i, t in enumerate(tracks)
    )
    prompt = (_PROMPT_WITH_GENRE if include_genre else _PROMPT_NO_GENRE).format(songs=songs_text)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def build_song_rows(
    tracks: list,
    estimates: list,
    start_id: int,
    override_genre: Optional[str] = None,
) -> List[dict]:
    """
    Combine raw Spotify track metadata with OpenAI feature estimates into CSV rows.

    override_genre: if set (e.g. from genre-based search), use it instead of the LLM estimate.
    """
    lookup = {
        (e.get("title", "").lower(), e.get("artist", "").lower()): e
        for e in estimates
    }
    rows = []
    for i, track in enumerate(tracks):
        title = track["name"]
        artist = track["artists"][0]["name"]
        est = lookup.get((title.lower(), artist.lower())) or (estimates[i] if i < len(estimates) else {})

        mood = est.get("mood", "chill")
        if mood not in VALID_MOODS:
            mood = "chill"

        genre = override_genre or est.get("genre", "pop")
        if genre not in GENRE_VALUES:
            genre = "pop"

        rows.append({
            "id":               start_id + len(rows),
            "title":            title,
            "artist":           artist,
            "genre":            genre,
            "mood":             mood,
            "energy":           round(float(est.get("energy", 0.5)), 2),
            "tempo_bpm":        int(est.get("tempo_bpm", 100)),
            "valence":          round(float(est.get("valence", 0.5)), 2),
            "danceability":     round(float(est.get("danceability", 0.5)), 2),
            "acousticness":     round(float(est.get("acousticness", 0.5)), 2),
            "popularity":       track.get("popularity", 50),
            "release_decade":   release_decade(track["album"]["release_date"]),
            "mood_tags":        est.get("mood_tags", ""),
            "live_energy":      round(float(est.get("live_energy", 0.5)), 2),
            "lyrical_depth":    round(float(est.get("lyrical_depth", 0.5)), 2),
            "instrumentalness": round(float(est.get("instrumentalness", 0.5)), 2),
            "spotify_id":       track["id"],
        })
    return rows
