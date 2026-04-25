"""
Populates data/songs.csv with real Spotify tracks.

Spotify restricts audio-features and recommendations for apps created after
Nov 2024, so this uses sp.search() with genre filters for track discovery and
OpenAI to estimate audio feature values (energy, valence, danceability, etc.).

Usage:
    python src/spotify_fetch.py                      # all 16 genres, 10 tracks each
    python src/spotify_fetch.py --genres pop rock --limit 20
    python src/spotify_fetch.py --output data/songs.csv --append
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.spotify_utils import (
    CSV_COLUMNS,
    GENRE_SEARCH_MAP,
    build_song_rows,
    estimate_features,
)

MAX_PER_PAGE = 10  # Spotify caps new-app search results at 10 per request


def _search_genre(sp, spotify_genre: str, limit: int) -> list:
    tracks = []
    offset = 0
    while len(tracks) < limit:
        batch = min(MAX_PER_PAGE, limit - len(tracks))
        try:
            results = sp.search(
                q=f'genre:"{spotify_genre}"',
                type="track",
                limit=batch,
                offset=offset,
            )
        except Exception as exc:
            print(f"  WARNING: search failed for '{spotify_genre}': {exc}", file=sys.stderr)
            break
        items = results["tracks"]["items"]
        if not items:
            break
        tracks.extend(items)
        offset += batch
    return tracks[:limit]


def fetch_songs(genres: list, limit: int) -> list:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from openai import OpenAI

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not client_id or not client_secret or client_id == "your_client_id_here":
        sys.exit("ERROR: Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
    if not openai_key:
        sys.exit("ERROR: Set OPENAI_API_KEY in .env")

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    ))
    openai_client = OpenAI(api_key=openai_key)

    all_rows = []
    for genre in genres:
        spotify_genre = GENRE_SEARCH_MAP.get(genre)
        if spotify_genre is None:
            print(f"  SKIP: unknown genre '{genre}'", file=sys.stderr)
            continue

        print(f"  [{genre}] Searching Spotify for genre:\"{spotify_genre}\"...")
        tracks = _search_genre(sp, spotify_genre, limit)
        if not tracks:
            continue
        print(f"    Found {len(tracks)} tracks. Estimating audio features with OpenAI...")

        estimates = estimate_features(tracks, openai_client, include_genre=False)
        rows = build_song_rows(tracks, estimates, start_id=len(all_rows) + 1, override_genre=genre)
        print(f"    -> {len(rows)} rows ready.")
        all_rows.extend(rows)

    # Re-number ids sequentially from 1
    for i, row in enumerate(all_rows, start=1):
        row["id"] = i

    return all_rows


def write_csv(rows: list, output: str, append: bool) -> None:
    path = Path(output)
    start_id = 1

    if append and path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        start_id = max((int(r["id"]) for r in existing), default=0) + 1
        for i, row in enumerate(rows):
            row["id"] = start_id + i
        mode, write_header = "a", False
    else:
        mode, write_header = "w", True

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    action = "Appended" if append and start_id > 1 else "Wrote"
    print(f"\n{action} {len(rows)} tracks to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Spotify tracks into songs.csv")
    parser.add_argument(
        "--genres", nargs="+",
        default=list(GENRE_SEARCH_MAP.keys()),
        help="Internal genre names to fetch (default: all 16)",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Tracks per genre, max 50 (default: 10)",
    )
    parser.add_argument(
        "--output", default="data/songs.csv",
        help="Output CSV path (default: data/songs.csv)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    print(f"Fetching Spotify data: {len(args.genres)} genres x {args.limit} tracks each")
    rows = fetch_songs(args.genres, args.limit)
    if not rows:
        sys.exit("No tracks fetched. Check credentials and genre names.")
    write_csv(rows, args.output, args.append)


if __name__ == "__main__":
    main()
