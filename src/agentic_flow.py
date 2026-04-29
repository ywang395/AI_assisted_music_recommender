from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from src.config import HISTORY_JSONL, SONGS_CSV, USER_PROFILE_JSON
from src.llm_reeval import (
    _MIN_EVENTS,
    deterministic_update,
    load_last_n_history,
    save_profile,
    update_profile_at_session_end,
)
from src.models import StableUserProfile


@dataclass
class AgenticFlowResult:
    songs: List[dict]
    profile: StableUserProfile
    reason: str
    history_count: int = 0
    candidate_changes: dict = field(default_factory=dict)


def run_startup_agentic_flow(
    stable: StableUserProfile,
    songs_csv: str = SONGS_CSV,
    history_path: str = HISTORY_JSONL,
    profile_path: str = USER_PROFILE_JSON,
) -> AgenticFlowResult:
    """Run the startup evidence -> profile -> recommendation preparation loop."""
    print("\n" + "=" * 64)
    print("  PIPELINE - Startup Agentic Flow")
    print("=" * 64)

    print("\n[A] Retrieving listening data from Spotify (3 sources)...")
    t0 = time.perf_counter()
    from src.spotify_sync import run_sync

    songs = run_sync(songs_csv, history_path)
    sync_elapsed = time.perf_counter() - t0
    print(f"    -> {len(songs)} songs in library | {sync_elapsed:.1f}s total")

    print("\n[B] Deterministic analysis of listening history...")
    t0 = time.perf_counter()
    history = load_last_n_history(history_path)
    if len(history) < _MIN_EVENTS:
        changes = {}
        det_elapsed = time.perf_counter() - t0
        print(
            f"    -> {len(history)} events analysed | 0 candidate changes | "
            f"{det_elapsed*1000:.0f}ms"
        )
        print(f"       evidence gate: need at least {_MIN_EVENTS} events before profile updates")
    else:
        changes, _, _ = deterministic_update(history, stable)
        det_elapsed = time.perf_counter() - t0
        print(f"    -> {len(history)} events analysed | {len(changes)} candidate changes | {det_elapsed*1000:.0f}ms")
        for field_name, value in changes.items():
            print(f"       candidate: {field_name} = {value!r}")

    print("\n[C] LLM refinement (OpenAI gpt-4.1-mini)...")
    t0 = time.perf_counter()
    if len(history) < _MIN_EVENTS:
        updated_stable, reason = stable, "no_change: insufficient history"
    else:
        updated_stable, reason = update_profile_at_session_end(history_path, stable)
    llm_elapsed = time.perf_counter() - t0
    if len(history) < _MIN_EVENTS:
        print(f"    -> skipped: insufficient history | {llm_elapsed:.1f}s")
    else:
        print(f"    -> method: {reason.split('|')[0].strip()} | {llm_elapsed:.1f}s")

    print("\n[D] Profile versioning...")
    if updated_stable.version != stable.version:
        save_profile(updated_stable, profile_path)
        print(f"    -> v{stable.version} -> v{updated_stable.version} saved")
        print(f"    -> reason: {reason}")
    else:
        print(f"    -> no update: {reason}")

    print("\n" + "=" * 64 + "\n")

    return AgenticFlowResult(
        songs=songs,
        profile=updated_stable,
        reason=reason,
        history_count=len(history),
        candidate_changes=changes,
    )
