"""
Command line runner for the Music Recommender Simulation.
"""

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
from recommender import SCORING_MODES, load_songs, recommend_songs
from src.config import HISTORY_JSONL, SONGS_CSV, USER_PROFILE_JSON


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _print_table(rows, headers) -> None:
    widths = []
    for index, header in enumerate(headers):
        cell_width = 0
        for row in rows:
            lines = str(row[index]).splitlines() or [""]
            cell_width = max(cell_width, max(len(line) for line in lines))
        widths.append(max(len(header), cell_width))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_row = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"

    print(border)
    print(header_row)
    print(border)
    for row in rows:
        split_cells = [str(cell).splitlines() or [""] for cell in row]
        row_height = max(len(cell_lines) for cell_lines in split_cells)
        for line_index in range(row_height):
            rendered = []
            for col_index, cell_lines in enumerate(split_cells):
                line = cell_lines[line_index] if line_index < len(cell_lines) else ""
                rendered.append(line.ljust(widths[col_index]))
            print("| " + " | ".join(rendered) + " |")
        print(border)


def _wrap_reasons(text: str, width: int = 56) -> str:
    parts = [part.strip() for part in text.split(",")]
    wrapped_lines = []
    for part in parts:
        wrapped_lines.extend(textwrap.wrap(part, width=width) or [""])
    return "\n".join(wrapped_lines)


def _format_reasons(text: str, width: int = 56, points_width: int = 8) -> str:
    parts = [part.strip() for part in text.split(",")]
    lines = []
    for part in parts:
        marker = " ("
        if marker in part and part.endswith(")"):
            label, points = part.rsplit(marker, 1)
            points = "(" + points
            label_width = max(8, width - points_width - 1)
            wrapped_label_lines = textwrap.wrap(label.strip(), width=label_width) or [""]
            for index, label_line in enumerate(wrapped_label_lines):
                if index == len(wrapped_label_lines) - 1:
                    lines.append(f"{label_line.ljust(label_width)} {points.rjust(points_width)}")
                else:
                    lines.append(label_line)
        else:
            lines.extend(textwrap.wrap(part, width=width) or [""])
    return "\n".join(lines)


def _default_stable_profile():
    from src.models import StableUserProfile

    return StableUserProfile(
        favorite_genre="pop",
        favorite_mood="sad",
        target_energy=0.1,
        likes_acoustic=True,
        target_danceability=0.1,
        target_valence=0.1,
        desired_popularity=0.85,
        preferred_decade=2010,
    )


def _profile_to_user_prefs(stable) -> dict:
    return {
        "genre": stable.favorite_genre,
        "artist": stable.favorite_artist,
        "mood": stable.favorite_mood,
        "energy": stable.target_energy,
        "tempo_bpm": stable.target_tempo_bpm,
        "danceability": stable.target_danceability,
        "valence": stable.target_valence,
        "likes_acoustic": stable.likes_acoustic,
        "popularity": stable.desired_popularity,
        "release_decade": stable.preferred_decade,
        "mood_tags": stable.preferred_mood_tags,
        "live_energy": stable.target_live_energy,
        "lyrical_depth": stable.target_lyrical_depth,
        "instrumentalness": stable.target_instrumentalness,
    }


def _top_titles(stable, songs, k: int = 5):
    recs = recommend_songs(_profile_to_user_prefs(stable), songs, k=k, mode=stable.scoring_mode)
    return [song["title"] for song, _, _ in recs]


def _run_profile_sync_preview(apply_changes: bool) -> int:
    from src.llm_reeval import load_profile, save_profile, update_profile_at_session_end

    songs = load_songs(SONGS_CSV)
    stable = load_profile(USER_PROFILE_JSON) or _default_stable_profile()
    new_stable, reason = update_profile_at_session_end(HISTORY_JSONL, stable)

    print("\n" + "=" * 88)
    print("PROFILE SYNC PREVIEW")
    print("=" * 88)
    print(f"version: {stable.version} -> {new_stable.version}")
    print(f"genre  : {stable.favorite_genre} -> {new_stable.favorite_genre}")
    print(f"mood   : {stable.favorite_mood} -> {new_stable.favorite_mood}")
    print(f"energy : {stable.target_energy} -> {new_stable.target_energy}")
    print(f"tempo  : {stable.target_tempo_bpm} -> {new_stable.target_tempo_bpm}")
    print(f"tags   : {stable.preferred_mood_tags} -> {new_stable.preferred_mood_tags}")
    print(f"reason : {reason}")

    before_titles = _top_titles(stable, songs)
    after_titles = _top_titles(new_stable, songs)
    print(f"top5 before: {before_titles}")
    print(f"top5 after : {after_titles}")

    if apply_changes:
        if new_stable.version != stable.version:
            save_profile(new_stable, USER_PROFILE_JSON)
            print(f"profile saved to {USER_PROFILE_JSON}")
        else:
            print("no profile changes were saved")

    return 0


def main() -> None:
    from src.llm_reeval import load_profile

    songs = load_songs(SONGS_CSV)
    stable = load_profile(USER_PROFILE_JSON) or _default_stable_profile()

    if "--sync-profile-preview" in sys.argv:
        raise SystemExit(_run_profile_sync_preview(apply_changes=False))

    if "--sync-profile" in sys.argv:
        raise SystemExit(_run_profile_sync_preview(apply_changes=True))

    if "--now-playing" in sys.argv:
        from src.player import run_now_playing

        # Sync Spotify history → update songs.csv + history.jsonl → refresh profile
        if "--no-sync" not in sys.argv:
            print("[startup] Syncing Spotify listening history...")
            from src.spotify_sync import run_sync
            songs = run_sync(SONGS_CSV, HISTORY_JSONL)

            # Re-evaluate profile against newly imported history before playing
            from src.llm_reeval import update_profile_at_session_end, save_profile
            updated_stable, reason = update_profile_at_session_end(HISTORY_JSONL, stable)
            if updated_stable.version != stable.version:
                save_profile(updated_stable, USER_PROFILE_JSON)
                print(f"[startup] Profile updated to v{updated_stable.version}: {reason}")
                stable = updated_stable
            else:
                print(f"[startup] Profile unchanged: {reason}")

        run_now_playing(songs, stable)
        return

    mode = stable.scoring_mode
    user_prefs = _profile_to_user_prefs(stable)

    recommendations = recommend_songs(user_prefs, songs, k=5, mode=mode)

    print("\n" + "=" * 88)
    print("TOP RECOMMENDATIONS")
    print(f"mode={mode} | available_modes={', '.join(SCORING_MODES.keys())}")
    print(
            "genre={genre} | mood={mood} | energy={energy} | tempo={tempo} | decade={decade} | tags={tags}".format(
                genre=user_prefs["genre"],
                mood=user_prefs["mood"],
                energy=user_prefs["energy"],
                tempo=user_prefs["tempo_bpm"],
                decade=user_prefs["release_decade"],
                tags="|".join(user_prefs["mood_tags"]),
            )
    )
    print("=" * 88)

    rows = []
    for index, (song, score, explanation) in enumerate(recommendations, start=1):
        rows.append(
            [
                index,
                _truncate(song["title"], 18),
                _truncate(song["artist"], 14),
                song["genre"],
                song["mood"],
                f"{score:.2f}",
                _format_reasons(explanation),
            ]
        )

    _print_table(rows, ["#", "Title", "Artist", "Genre", "Mood", "Score", "Reasons"])


if __name__ == "__main__":
    main()
