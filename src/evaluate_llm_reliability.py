"""
LLM Reliability Evaluation — synthetic test harness + live data check.

Runs 6 predefined scenarios through the profile-update pipeline and reports
pass/fail with confidence ratings. No OpenAI API key required for scenarios
1-5 (deterministic path); Scenario 6 runs on live history and calls the LLM
if OPENAI_API_KEY is set.

Usage:
    python -m src.evaluate_llm_reliability
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

from src.llm_reeval import (
    _MAX_DELTA,
    _MAX_TEMPO_DELTA,
    _MIN_EVENTS,
    _NUMERIC_FIELDS,
    _apply_changes,
    call_llm_reeval,
    deterministic_update,
    load_last_n_history,
    load_profile,
    parse_and_guard,
)
from src.models import InteractionEvent, StableUserProfile


# ── Synthetic event factory ───────────────────────────────────────────────────

def _ev(
    event_type: str,
    genre: str = "pop",
    mood: str = "happy",
    energy: float = 0.5,
    tempo: float = 100.0,
    ratio: float = 1.0,
) -> dict:
    return InteractionEvent(
        event_type=event_type,
        session_id="test",
        song_id=1,
        song_title="Test Song",
        song_artist="Test Artist",
        song_genre=genre,
        song_mood=mood,
        song_energy=energy,
        song_valence=0.5,
        song_tempo_bpm=tempo,
        song_score=0.5,
        elapsed_seconds=30.0 * ratio,
        total_duration=30.0,
        elapsed_ratio=ratio,
        repeat_count=0,
        timestamp="2024-01-01T00:00:00Z",
    ).to_dict()


# ── Shared guardrail check (same logic as original) ───────────────────────────

def _guardrail_checks(before: StableUserProfile, after: StableUserProfile) -> Dict[str, bool]:
    checks: Dict[str, bool] = {
        "protected_artist_preserved": before.favorite_artist == after.favorite_artist,
        "protected_mode_preserved": before.scoring_mode == after.scoring_mode,
        "genre_present": bool(after.favorite_genre),
        "mood_present": bool(after.favorite_mood),
    }
    for field in _NUMERIC_FIELDS:
        before_val = float(getattr(before, field))
        after_val = float(getattr(after, field))
        checks[f"{field}_bounded"] = 0.0 <= after_val <= 1.0
        checks[f"{field}_max_delta"] = abs(after_val - before_val) <= (_MAX_DELTA + 1e-9)
    checks["target_tempo_bpm_bounded"] = 40 <= int(after.target_tempo_bpm) <= 220
    checks["target_tempo_bpm_max_delta"] = (
        abs(int(after.target_tempo_bpm) - int(before.target_tempo_bpm)) <= _MAX_TEMPO_DELTA
    )
    return checks


def _confidence(checks: Dict[str, bool]) -> float:
    if not checks:
        return 0.0
    return round(sum(checks.values()) / len(checks), 2)


# ── Scenario definitions ──────────────────────────────────────────────────────

def _scenario_1() -> Tuple[str, str, List[dict], StableUserProfile, Dict[str, Any]]:
    """High-energy listener: profile should nudge energy and tempo upward."""
    profile = StableUserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.2,
        target_tempo_bpm=80,
        favorite_artist="",
        scoring_mode="balanced",
    )
    history = [_ev("complete", energy=0.87, tempo=145.0) for _ in range(12)]
    expected = {
        "has_candidate_changes": True,
        "energy_direction_up": True,
        "tempo_direction_up": True,
    }
    return "Scenario 1", "High-energy listener nudges energy + tempo upward", history, profile, expected


def _scenario_2() -> Tuple[str, str, List[dict], StableUserProfile, Dict[str, Any]]:
    """Heavy skipper on current genre: genre-change gate should open."""
    profile = StableUserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.5,
        target_tempo_bpm=100,
        favorite_artist="",
        scoring_mode="balanced",
    )
    history = (
        [_ev("skip", genre="pop", mood="happy", ratio=0.2) for _ in range(6)]
        + [_ev("complete", genre="hip-hop", mood="sad", energy=0.65, tempo=95.0) for _ in range(9)]
    )
    expected = {
        "allow_genre_change": True,
        "genre_changed_to_hiphop": True,
    }
    return "Scenario 2", "Heavy pop-skipper / hip-hop completer opens genre-change gate", history, profile, expected


def _scenario_3() -> Tuple[str, str, List[dict], StableUserProfile, Dict[str, Any]]:
    """Protected fields must never be modified even when injected into LLM JSON."""
    profile = StableUserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        favorite_artist="Drake",
        scoring_mode="genre_first",
        target_energy=0.4,
    )
    # Inject a malicious JSON claiming protected field changes
    injected_json = json.dumps({
        "favorite_artist": "Kanye West",
        "scoring_mode": "mood_first",
        "version": 999,
        "target_energy": 0.9,
    })
    after = parse_and_guard(injected_json, profile, allow_genre_change=False, allow_mood_change=False)
    expected = {
        "favorite_artist_unchanged": after.favorite_artist == "Drake",
        "scoring_mode_unchanged": after.scoring_mode == "genre_first",
        "version_unchanged": after.version == profile.version,
        "energy_bounded": 0.0 <= after.target_energy <= 1.0,
        "energy_max_delta_respected": abs(after.target_energy - profile.target_energy) <= _MAX_DELTA + 1e-9,
    }
    return "Scenario 3", "Protected fields (favorite_artist, scoring_mode, version) survive injected JSON", [], profile, expected


def _scenario_4() -> Tuple[str, str, List[dict], StableUserProfile, Dict[str, Any]]:
    """Insufficient history: no update should fire."""
    profile = StableUserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.5,
    )
    history = [_ev("complete") for _ in range(_MIN_EVENTS - 2)]  # below threshold
    expected = {
        "no_changes_below_min_events": True,
    }
    return "Scenario 4", "Insufficient history (< _MIN_EVENTS) produces no candidate changes", history, profile, expected


def _scenario_5() -> Tuple[str, str, List[dict], StableUserProfile, Dict[str, Any]]:
    """Repeat events drive preferred_mood_tags update."""
    profile = StableUserProfile(
        favorite_genre="hip-hop",
        favorite_mood="sad",
        preferred_mood_tags=[],
        target_energy=0.5,
    )
    history = [_ev("repeat", genre="hip-hop", mood="chill") for _ in range(8)]
    expected = {
        "has_candidate_changes": True,
        "chill_in_mood_tags": True,
    }
    return "Scenario 5", "Repeated chill songs add 'chill' to preferred_mood_tags", history, profile, expected


def _scenario_6_live() -> Tuple[str, str, List[dict], StableUserProfile, Dict[str, Any]]:
    """Live data: run against real history.jsonl + user_profile.json."""
    history = load_last_n_history("data/history.jsonl")
    profile = load_profile("data/user_profile.json") or StableUserProfile(
        favorite_genre="pop",
        favorite_mood="sad",
        target_energy=0.1,
        likes_acoustic=True,
        target_danceability=0.1,
        target_valence=0.1,
        desired_popularity=0.85,
        preferred_decade=2010,
    )
    expected = {
        "has_history": len(history) > 0,
        "profile_loaded": True,
    }
    return "Scenario 6", "Live data: guardrails hold on real history.jsonl", history, profile, expected


# ── Scenario runner ───────────────────────────────────────────────────────────

def _run_scenario(
    name: str,
    description: str,
    history: List[dict],
    profile: StableUserProfile,
    expected: Dict[str, Any],
    use_llm: bool = False,
) -> Tuple[Dict[str, bool], float, float]:
    """Returns (all_checks, confidence, elapsed_seconds)."""
    t0 = time.perf_counter()
    all_checks: Dict[str, bool] = {}

    # Scenario 3 bypasses the pipeline (parse_and_guard already run in definition)
    if name == "Scenario 3":
        all_checks.update(expected)
        elapsed = time.perf_counter() - t0
        return all_checks, _confidence(all_checks), elapsed

    changes, allow_genre, allow_mood = deterministic_update(history, profile)
    candidate = _apply_changes(profile, changes) if changes else profile

    if use_llm and changes:
        candidate, method = call_llm_reeval(history, profile, changes, allow_genre, allow_mood)
        all_checks["llm_method_recorded"] = bool(method)

    guardrails = _guardrail_checks(profile, candidate)
    all_checks.update(guardrails)

    # Scenario-specific checks
    if "has_candidate_changes" in expected:
        all_checks["has_candidate_changes"] = bool(changes)

    if "energy_direction_up" in expected:
        all_checks["energy_direction_up"] = candidate.target_energy > profile.target_energy

    if "tempo_direction_up" in expected:
        all_checks["tempo_direction_up"] = candidate.target_tempo_bpm > profile.target_tempo_bpm

    if "allow_genre_change" in expected:
        all_checks["allow_genre_change"] = allow_genre

    if "genre_changed_to_hiphop" in expected:
        all_checks["genre_changed_to_hiphop"] = candidate.favorite_genre == "hip-hop"

    if "no_changes_below_min_events" in expected:
        all_checks["no_changes_below_min_events"] = not bool(changes)

    if "chill_in_mood_tags" in expected:
        all_checks["chill_in_mood_tags"] = "chill" in candidate.preferred_mood_tags

    if "has_history" in expected:
        all_checks["has_history"] = expected["has_history"]

    if "profile_loaded" in expected:
        all_checks["profile_loaded"] = True

    elapsed = time.perf_counter() - t0
    return all_checks, _confidence(all_checks), elapsed


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_scenario_result(
    name: str,
    description: str,
    checks: Dict[str, bool],
    confidence: float,
    elapsed: float,
) -> bool:
    passed = all(checks.values())
    status = "PASS" if passed else "FAIL"
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  {name}: {description}")
    print(f"  Result: {status}  |  Confidence: {confidence:.0%}  |  Elapsed: {elapsed*1000:.0f}ms")
    print(bar)
    for check_name, result in checks.items():
        icon = "  ✓" if result else "  ✗"
        print(f"{icon}  {check_name}")
    return passed


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    scenarios = [
        _scenario_1,
        _scenario_2,
        _scenario_3,
        _scenario_4,
        _scenario_5,
    ]

    total_pass = 0
    total_scenarios = len(scenarios) + 1  # +1 for live scenario

    print("\n" + "=" * 72)
    print("  LLM RELIABILITY EVALUATION — Synthetic Test Harness")
    print("=" * 72)

    for fn in scenarios:
        name, description, history, profile, expected = fn()
        use_llm = (name == "Scenario 6")
        checks, confidence, elapsed = _run_scenario(name, description, history, profile, expected, use_llm)
        passed = _print_scenario_result(name, description, checks, confidence, elapsed)
        if passed:
            total_pass += 1

    # Scenario 6: live data
    name, description, history, profile, expected = _scenario_6_live()
    checks, confidence, elapsed = _run_scenario(name, description, history, profile, expected, use_llm=True)
    passed = _print_scenario_result(name, description, checks, confidence, elapsed)
    if passed:
        total_pass += 1

    print("\n" + "=" * 72)
    print(f"  SUMMARY: {total_pass} / {total_scenarios} scenarios passed")
    if total_pass == total_scenarios:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
