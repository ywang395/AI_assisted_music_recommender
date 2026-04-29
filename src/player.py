from __future__ import annotations

import os
import queue
import select
import shutil
import sys
import time
import threading
import textwrap
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import uuid4

from src.models import InteractionEvent, SessionState, StableUserProfile
from src.recommender import recommend_songs

# Maximum nudge magnitude applied per session
_NUDGE_MAX = 0.25
_NUDGE_STEP = 0.05
_TICK = 0.5  # seconds between UI redraws


# ── Duration ─────────────────────────────────────────────────────────────────

def compute_song_duration(song: dict) -> float:
    bpm = song.get("tempo_bpm", 100)
    if bpm < 80:
        duration = 35.0
    elif bpm > 140:
        duration = 25.0
    else:
        duration = 30.0
    return max(15.0, min(45.0, duration))


# ── Preference merging ────────────────────────────────────────────────────────

def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def merge_prefs(stable: StableUserProfile, session: SessionState) -> dict:
    return {
        "genre":            stable.favorite_genre,
        "artist":           stable.favorite_artist,
        "mood":             stable.favorite_mood,
        "energy":           _clamp(stable.target_energy + session.energy_nudge),
        "tempo_bpm":        stable.target_tempo_bpm,
        "likes_acoustic":   stable.likes_acoustic,
        "danceability":     _clamp(stable.target_danceability + session.danceability_nudge),
        "valence":          _clamp(stable.target_valence + session.valence_nudge),
        "popularity":       stable.desired_popularity,
        "release_decade":   stable.preferred_decade,
        "mood_tags":        stable.preferred_mood_tags,
        "live_energy":      stable.target_live_energy,
        "lyrical_depth":    stable.target_lyrical_depth,
        "instrumentalness": stable.target_instrumentalness,
        "song_penalties": {
            str(song_id): min(0.75, 0.55 + (0.1 * session.recent_skips.count(song_id)))
            for song_id in session.recent_skips
        },
    }


def get_fresh_queue(
    songs: List[dict],
    stable: StableUserProfile,
    session: SessionState,
    k: int = 8,
) -> List[Tuple[dict, float, str]]:
    candidates = [s for s in songs if s["id"] not in session.played_ids]
    if not candidates:
        return []
    prefs = merge_prefs(stable, session)
    results = recommend_songs(prefs, candidates, k=min(k, len(candidates)), mode=stable.scoring_mode)
    return list(results)


# ── Session nudge ─────────────────────────────────────────────────────────────

def apply_session_nudge(session: SessionState, event: InteractionEvent) -> None:
    if event.event_type not in ("skip", "repeat"):
        return

    direction = -1.0 if event.event_type == "skip" else 1.0

    def _nudge(current: float, song_val: float) -> float:
        delta = direction * _NUDGE_STEP * (song_val - 0.5) * 2
        return max(-_NUDGE_MAX, min(_NUDGE_MAX, current + delta))

    session.energy_nudge = _nudge(session.energy_nudge, event.song_energy)
    session.valence_nudge = _nudge(session.valence_nudge, event.song_valence)


# ── Keyboard listener ─────────────────────────────────────────────────────────

class KeyboardListener(threading.Thread):
    _DEBOUNCE_SECONDS = 0.25

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._key_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._last_emitted_key: Optional[str] = None
        self._last_emitted_at: float = 0.0

    def run(self) -> None:
        if sys.platform == "win32":
            return
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":
                        r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r2:
                            rest = sys.stdin.read(2)
                            if rest == "[C":
                                ch = "RIGHT"
                    if ch in {"s", "r", "q", "RIGHT"}:
                        now = time.monotonic()
                        if ch == self._last_emitted_key and (now - self._last_emitted_at) < self._DEBOUNCE_SECONDS:
                            continue
                        self._last_emitted_key = ch
                        self._last_emitted_at = now
                    self._key_queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def stop(self) -> None:
        self._stop_event.set()

    def get_key(self) -> Optional[str]:
        try:
            return self._key_queue.get_nowait()
        except queue.Empty:
            return None

    def clear_pending(self) -> None:
        while True:
            try:
                self._key_queue.get_nowait()
            except queue.Empty:
                self._last_emitted_key = None
                self._last_emitted_at = 0.0
                return


# ── Terminal UI ───────────────────────────────────────────────────────────────

class NowPlayingUI:
    _DEFAULT_WIDTH = 64
    _MIN_WIDTH = 48

    def __init__(self) -> None:
        self._last_song_id: Optional[int] = None

    def _width(self) -> int:
        cols = shutil.get_terminal_size(fallback=(self._DEFAULT_WIDTH, 24)).columns
        return max(self._MIN_WIDTH, min(self._DEFAULT_WIDTH, cols - 2))

    @staticmethod
    def _bar(elapsed: float, total: float, width: int = 20) -> str:
        ratio = min(1.0, elapsed / total) if total > 0 else 0.0
        filled = int(ratio * width)
        bar = "=" * filled + (">" if filled < width else "") + " " * max(0, width - filled - 1)
        return f"[{bar}]"

    @staticmethod
    def _trunc(text: str, max_len: int) -> str:
        if max_len <= 3:
            return text[:max_len]
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _fit_line(self, text: str, width: int) -> str:
        return self._trunc(text, width).ljust(width)

    def update_progress(self, elapsed: float, total: float) -> None:
        w = self._width()
        bar_width = min(20, max(10, w - 22))
        bar = self._bar(elapsed, total, width=bar_width)
        elapsed_str = f"{int(elapsed)}s / {int(total)}s"
        progress_line = self._fit_line(f"  {bar}  {elapsed_str}", w)
        sys.stdout.write("\r" + progress_line)
        sys.stdout.flush()

    def show_song(
        self,
        song: dict,
        score: float,
        explanation: str,
        status_msg: str = "",
        profile_version: int = 1,
        recompute_count: int = 0,
    ) -> None:
        w = self._width()
        sep = "=" * w
        thin = "-" * w

        header_right = f"[queue #{recompute_count} | v{profile_version}]"
        header_left = "  NOW PLAYING"
        header_pad = w - len(header_left) - len(header_right)
        header = self._fit_line(header_left + " " * max(1, header_pad) + header_right, w)

        controls = "  s / RIGHT Skip    r Repeat    q Quit"
        if status_msg:
            bottom = self._fit_line(f"  {self._trunc(status_msg, w - 2)}", w)
        else:
            bottom = self._fit_line(controls, w)

        title_line = self._fit_line(f"  Title   : {song.get('title', '')}", w)
        artist_line = self._fit_line(f"  Artist  : {song.get('artist', '')}", w)
        meta_line = self._fit_line(
            f"  Genre   : {song.get('genre', '')} | Mood: {song.get('mood', '')} | BPM: {int(song.get('tempo_bpm', 0))}",
            w,
        )
        score_line = self._fit_line(
            f"  Energy  : {song.get('energy', 0):.2f} | Valence: {song.get('valence', 0):.2f} | Score: {score:.2f}",
            w,
        )

        lines = [
            sep,
            header,
            sep,
            title_line,
            artist_line,
            meta_line,
            score_line,
            sep,
            bottom,
            sep,
        ]

        if self._last_song_id is not None:
            sys.stdout.write("\r\n")

        output = "\r\n".join(lines) + "\r\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        self._last_song_id = song.get("id")

    def show_message(self, msg: str) -> None:
        line = f"  {msg}"
        sys.stdout.write("\r" + line + "\r\n")
        sys.stdout.flush()
        self._last_song_id = None

    def show_wrapped_message(self, msg: str) -> None:
        w = self._width()
        for line in textwrap.wrap(msg, width=max(24, w - 2)) or [""]:
            sys.stdout.write("  " + line + "\r\n")
        sys.stdout.flush()
        self._last_song_id = None

    def cleanup(self) -> None:
        sys.stdout.write("\r\n")
        sys.stdout.flush()
        self._last_song_id = None


# ── Main entry point ──────────────────────────────────────────────────────────

def run_now_playing(
    songs: List[dict],
    stable: StableUserProfile,
    history_path: str = "data/history.jsonl",
) -> None:
    if sys.platform == "win32":
        print("Now Playing mode requires macOS or Linux.")
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "Now Playing mode requires an interactive terminal (TTY). "
            "Run it in a real terminal, not a debug/output console."
        )
        return

    # Import here to keep player.py independent when llm_reeval isn't installed
    from src.llm_reeval import (
        append_events_to_history,
        update_profile_at_session_end,
        save_profile,
    )

    session = SessionState(
        session_id=str(uuid4()),
        stable_profile=stable,
    )
    session_events: List[InteractionEvent] = []

    queue_list = get_fresh_queue(songs, stable, session)
    listener = KeyboardListener()
    listener.start()
    ui = NowPlayingUI()

    def _shutdown_terminal() -> None:
        listener.stop()
        if listener.is_alive():
            listener.join(timeout=0.5)
        ui.cleanup()

    def _make_event(
        event_type: str,
        song: dict,
        score: float,
        elapsed: float,
        total: float,
        repeat_count: int = 0,
    ) -> InteractionEvent:
        return InteractionEvent(
            event_type=event_type,
            session_id=session.session_id,
            song_id=song["id"],
            song_title=song.get("title", ""),
            song_artist=song.get("artist", ""),
            song_genre=song.get("genre", ""),
            song_mood=song.get("mood", ""),
            song_energy=float(song.get("energy", 0.0)),
            song_valence=float(song.get("valence", 0.0)),
            song_tempo_bpm=float(song.get("tempo_bpm", 0.0)),
            song_score=score,
            elapsed_seconds=elapsed,
            total_duration=total,
            elapsed_ratio=elapsed / total if total > 0 else 0.0,
            repeat_count=repeat_count,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _finalize_session(song: dict, score: float, elapsed: float, total: float, reason_prefix: str) -> None:
        ui.show_song(
            song,
            score,
            "",
            status_msg="Saving history & updating profile...",
            profile_version=stable.version,
            recompute_count=session.recompute_count,
        )
        ui.update_progress(elapsed, total)
        append_events_to_history(session_events, history_path)
        new_stable, reason = update_profile_at_session_end(history_path, stable)
        if new_stable.version != stable.version:
            save_profile(new_stable, "data/user_profile.json")
            msg = f"{reason_prefix} Profile v{new_stable.version} saved. {reason}"
            status_msg = f"{reason_prefix} Profile v{new_stable.version} saved. See update reason below."
        else:
            msg = f"{reason_prefix} No profile changes saved. {reason}"
            status_msg = f"{reason_prefix} No profile changes saved. See reason below."

        ui.show_song(
            song,
            score,
            "",
            status_msg=status_msg,
            profile_version=new_stable.version,
            recompute_count=session.recompute_count,
        )
        ui.update_progress(elapsed, total)
        sys.stdout.write("\r\n")
        ui.show_wrapped_message(msg)
        time.sleep(2)
        _shutdown_terminal()

    def _on_quit(song: dict, score: float, elapsed: float, total: float) -> None:
        # quit = session termination only — not a dislike signal, not fed to nudge
        event = _make_event("quit", song, score, elapsed, total)
        session_events.append(event)
        _finalize_session(song, score, elapsed, total, "")

    try:
        while True:
            if not queue_list:
                if not session.played_ids:
                    _shutdown_terminal()
                    return
                last_song_id = session.played_ids[-1]
                last_song = next((s for s in songs if s["id"] == last_song_id), songs[0])
                _finalize_session(
                    last_song,
                    0.0,
                    compute_song_duration(last_song),
                    compute_song_duration(last_song),
                    "Session complete.",
                )
                return

            song, score, explanation = queue_list.pop(0)
            session.played_ids.append(song["id"])
            duration = compute_song_duration(song)
            start = time.monotonic()
            repeat_count = 0
            ui.show_song(
                song,
                score,
                explanation,
                profile_version=stable.version,
                recompute_count=session.recompute_count,
            )

            while True:
                elapsed = time.monotonic() - start
                ui.update_progress(elapsed, duration)

                key = listener.get_key()

                if key == "q":
                    _on_quit(song, score, elapsed, duration)
                    return

                elif key in ("s", "RIGHT"):
                    event = _make_event("skip", song, score, elapsed, duration)
                    session_events.append(event)
                    apply_session_nudge(session, event)
                    session.skips_since_last_recompute += 1
                    session.recent_skips.append(song["id"])
                    listener.clear_pending()
                    queue_list = get_fresh_queue(songs, stable, session)
                    session.skips_since_last_recompute = 0
                    session.recompute_count += 1
                    break

                elif key == "r":
                    repeat_count += 1
                    event = _make_event("repeat", song, score, elapsed, duration, repeat_count)
                    session_events.append(event)
                    apply_session_nudge(session, event)
                    session.recent_repeats.append(song["id"])
                    listener.clear_pending()
                    queue_list = get_fresh_queue(songs, stable, session)
                    session.recompute_count += 1
                    start = time.monotonic()
                    # stay on same song — continue inner loop

                elif elapsed >= duration:
                    event = _make_event("complete", song, score, elapsed, duration)
                    session_events.append(event)
                    session.recent_completes.append(song["id"])
                    break

                time.sleep(_TICK)

    except KeyboardInterrupt:
        # Best-effort save on Ctrl+C
        try:
            append_events_to_history(session_events, history_path)
            new_stable, _ = update_profile_at_session_end(history_path, stable)
            if new_stable.version != stable.version:
                save_profile(new_stable, "data/user_profile.json")
        except Exception:
            pass
        finally:
            _shutdown_terminal()
