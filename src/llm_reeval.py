from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env if present

from src.models import InteractionEvent, StableUserProfile

HISTORY_LIMIT = 75
HISTORY_PATH = "data/history.jsonl"
LLM_EVENT_LIMIT = 20

_VALID_SCORING_MODES = {"balanced", "genre_first", "mood_first", "energy_focused"}
_NUMERIC_FIELDS = (
    "target_energy",
    "target_danceability",
    "target_valence",
    "target_live_energy",
    "target_lyrical_depth",
    "target_instrumentalness",
    "desired_popularity",
)
_MAX_DELTA = 0.3
_MAX_TEMPO_DELTA = 30
_TEMPO_MIN = 40
_TEMPO_MAX = 220
_MIN_EVENTS = 5
_ENERGY_CHANGE_THRESHOLD = 0.05
_CATEGORY_EVIDENCE_THRESHOLD = 0.35

SYSTEM_PROMPT = """You are a music taste profiling assistant.

You will receive:
1. A user's current StableUserProfile as JSON
2. Candidate changes derived from deterministic analysis of their listening history
3. Up to 20 selected interaction events (skip / repeat / complete)

Your job is to refine the candidate changes — not invent new ones.
You may reduce or remove a proposed change if the evidence looks weak.
You may NOT propose changes to fields that are not already in candidate_changes,
EXCEPT you may override a deterministic genre/mood change if you judge the
evidence to be insufficient.

Fields you must NEVER modify (even if not in candidate_changes):
  favorite_artist, scoring_mode, version, last_updated, update_reason, previous_version

Return ONLY a valid JSON object containing the same keys as candidate_changes
(or a subset). Do not include any explanation or fields not in candidate_changes.
If you think no changes are warranted, return an empty JSON object: {}

Important: consider skipped songs as negative preference evidence, especially
when the user skips early in the track.
"""


# ── Sanitization ──────────────────────────────────────────────────────────────

def _sanitize(value: str, max_len: int = 80) -> str:
    return value.replace("\n", " ").replace("\r", " ").replace("\x00", "")[:max_len]


def _profile_to_user_prefs(profile: StableUserProfile) -> dict:
    return {
        "genre": profile.favorite_genre,
        "artist": profile.favorite_artist,
        "mood": profile.favorite_mood,
        "energy": profile.target_energy,
        "tempo_bpm": profile.target_tempo_bpm,
        "danceability": profile.target_danceability,
        "valence": profile.target_valence,
        "likes_acoustic": profile.likes_acoustic,
        "popularity": profile.desired_popularity,
        "release_decade": profile.preferred_decade,
        "mood_tags": profile.preferred_mood_tags,
        "live_energy": profile.target_live_energy,
        "lyrical_depth": profile.target_lyrical_depth,
        "instrumentalness": profile.target_instrumentalness,
    }


def summarize_recommendation_shift(
    before: StableUserProfile,
    after: StableUserProfile,
    songs_path: str = "data/songs.csv",
    k: int = 3,
) -> Tuple[str, bool]:
    from src.recommender import load_songs, recommend_songs

    songs = load_songs(songs_path)
    before_recs = recommend_songs(_profile_to_user_prefs(before), songs, k=k, mode=before.scoring_mode)
    after_recs = recommend_songs(_profile_to_user_prefs(after), songs, k=k, mode=after.scoring_mode)

    before_titles = [song["title"] for song, _, _ in before_recs]
    after_titles = [song["title"] for song, _, _ in after_recs]
    changed = before_titles != after_titles
    if changed:
        return f"top{k} changed: {before_titles[0]} -> {after_titles[0]}", True
    return f"top{k} unchanged: {', '.join(after_titles)}", False


# ── History I/O ───────────────────────────────────────────────────────────────

def load_last_n_history(path: str = HISTORY_PATH, n: int = HISTORY_LIMIT) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    recent = lines[-n:]
    events = []
    for line in recent:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def append_events_to_history(
    events: List[InteractionEvent],
    path: str = HISTORY_PATH,
    limit: int = HISTORY_LIMIT,
) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.to_dict()) + "\n")

    # Trim to limit
    with open(path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    if len(lines) > limit:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[-limit:])


# ── Profile persistence ───────────────────────────────────────────────────────

def save_profile(profile: StableUserProfile, path: str = "data/user_profile.json") -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)


def load_profile(path: str = "data/user_profile.json") -> Optional[StableUserProfile]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StableUserProfile(
            favorite_genre=data["favorite_genre"],
            favorite_mood=data["favorite_mood"],
            favorite_artist=data.get("favorite_artist", ""),
            scoring_mode=data.get("scoring_mode", "balanced"),
            target_energy=float(data.get("target_energy", 0.5)),
            target_danceability=float(data.get("target_danceability", 0.5)),
            target_valence=float(data.get("target_valence", 0.5)),
            target_live_energy=float(data.get("target_live_energy", 0.5)),
            target_lyrical_depth=float(data.get("target_lyrical_depth", 0.5)),
            target_instrumentalness=float(data.get("target_instrumentalness", 0.5)),
            target_tempo_bpm=int(data.get("target_tempo_bpm", 100)),
            desired_popularity=float(data.get("desired_popularity", 0.5)),
            preferred_decade=int(data.get("preferred_decade", 2010)),
            likes_acoustic=bool(data.get("likes_acoustic", False)),
            preferred_mood_tags=list(data.get("preferred_mood_tags", [])),
            version=int(data.get("version", 1)),
            last_updated=data.get("last_updated", ""),
            update_reason=data.get("update_reason", ""),
            previous_version=dict(data.get("previous_version", {})),
        )
    except (KeyError, ValueError, TypeError):
        return None


# ── Deterministic update ──────────────────────────────────────────────────────

def deterministic_update(
    history: List[dict],
    stable: StableUserProfile,
) -> Tuple[dict, bool, bool]:
    """Return (candidate_changes, allow_genre_change, allow_mood_change).

    candidate_changes maps field_name -> proposed_new_value.
    Fields absent from candidate_changes are not proposed for change.
    """
    if not history:
        return {}, False, False

    # Separate events by type (quit is neutral — excluded from preference signals)
    positive = [e for e in history if e.get("event_type") in ("complete", "repeat")]
    negative = [e for e in history if e.get("event_type") == "skip"]

    changes: dict = {}

    # ── Numeric fields: nudge toward mean of positive events ──────────────────
    # Only energy and tempo are carried in InteractionEvent; others need genre proxy
    if positive:
        mean_energy = sum(e["song_energy"] for e in positive) / len(positive)
        current = stable.target_energy
        if abs(mean_energy - current) > _ENERGY_CHANGE_THRESHOLD:
            nudge = 0.15 * (1 if mean_energy > current else -1)
            changes["target_energy"] = round(max(0.0, min(1.0, current + nudge)), 3)

        mean_tempo = sum(float(e.get("song_tempo_bpm", 100.0)) for e in positive) / len(positive)
        current_tempo = float(stable.target_tempo_bpm)
        if abs(mean_tempo - current_tempo) > 10:
            direction = 1 if mean_tempo > current_tempo else -1
            changes["target_tempo_bpm"] = int(max(_TEMPO_MIN, min(_TEMPO_MAX, current_tempo + (20 * direction))))

    # ── Preferred decade ──────────────────────────────────────────────────────
    # (event schema doesn't carry decade — skip for now; future: add song_decade to event)

    # ── Preferred mood tags ───────────────────────────────────────────────────
    repeat_events = [e for e in history if e.get("event_type") == "repeat"]
    early_skip_moods = {
        e.get("song_mood", "")
        for e in negative
        if e.get("elapsed_ratio", 1.0) < 0.3
    }
    repeat_moods = [e.get("song_mood", "") for e in repeat_events if e.get("song_mood")]
    if repeat_moods:
        new_tags = list({m for m in repeat_moods if m not in early_skip_moods})
        if set(new_tags) != set(stable.preferred_mood_tags):
            changes["preferred_mood_tags"] = new_tags

    # ── Genre/mood change gates (relative ratios, not absolute fractions) ─────
    allow_genre_change = False
    allow_mood_change = False

    if negative and positive:
        skip_genres = Counter(e.get("song_genre", "") for e in negative)
        complete_genres = Counter(e.get("song_genre", "") for e in positive)
        top_complete = complete_genres.most_common(1)[0] if complete_genres else (None, 0)
        complete_ratio_top = top_complete[1] / max(len(positive), 1)
        disliked_genres = {genre for genre, count in skip_genres.items() if count >= 3}

        if (
            top_complete[0]
            and complete_ratio_top >= _CATEGORY_EVIDENCE_THRESHOLD
            and top_complete[0] != stable.favorite_genre
            and top_complete[0] not in disliked_genres
        ):
            allow_genre_change = True
            changes["favorite_genre"] = top_complete[0]

        current_mood = stable.favorite_mood
        skip_moods = Counter(e.get("song_mood", "") for e in negative)
        complete_moods = Counter(e.get("song_mood", "") for e in positive)

        skip_ratio_mood = skip_moods.get(current_mood, 0) / max(len(negative), 1)
        top_complete_mood = complete_moods.most_common(1)[0] if complete_moods else (None, 0)
        complete_ratio_mood = top_complete_mood[1] / max(len(positive), 1)

        if (
            skip_ratio_mood >= _CATEGORY_EVIDENCE_THRESHOLD
            and complete_ratio_mood >= _CATEGORY_EVIDENCE_THRESHOLD
            and top_complete_mood[0] != current_mood
        ):
            allow_mood_change = True
            changes["favorite_mood"] = top_complete_mood[0]

    return changes, allow_genre_change, allow_mood_change


# ── Retrieval ─────────────────────────────────────────────────────────────────

def select_relevant_history_for_llm(
    history: List[dict],
    limit: int = LLM_EVENT_LIMIT,
) -> List[dict]:
    """Select a compact, relevant evidence window for the LLM.

    Retrieval priority:
    - most recent early skips (strongest negative evidence)
    - most recent repeats (strong positive evidence)
    - most recent completes
    - preserve recency order in final output
    """
    relevant = [
        (index, event)
        for index, event in enumerate(history)
        if event.get("event_type") in {"skip", "repeat", "complete"}
    ]
    if len(relevant) <= limit:
        return [event for _, event in relevant]

    early_skips = [(i, e) for i, e in relevant if e.get("event_type") == "skip" and e.get("elapsed_ratio", 1.0) < 0.3]
    repeats = [(i, e) for i, e in relevant if e.get("event_type") == "repeat"]
    completes = [(i, e) for i, e in relevant if e.get("event_type") == "complete"]
    late_skips = [(i, e) for i, e in relevant if e.get("event_type") == "skip" and e.get("elapsed_ratio", 1.0) >= 0.3]

    selected: List[tuple[int, dict]] = []
    seen_positions = set()

    def add_recent(events: List[tuple[int, dict]], count: int) -> None:
        for position, event in reversed(events[-count:]):
            if position in seen_positions:
                continue
            selected.append((position, event))
            seen_positions.add(position)
            if len(selected) >= limit:
                return

    add_recent(early_skips, min(8, limit))
    if len(selected) < limit:
        add_recent(repeats, min(4, limit - len(selected)))
    if len(selected) < limit:
        add_recent(completes, min(8, limit - len(selected)))
    if len(selected) < limit:
        add_recent(late_skips, limit - len(selected))
    if len(selected) < limit:
        add_recent(relevant, limit - len(selected))

    selected = selected[:limit]
    selected.sort(key=lambda item: item[0])
    return [event for _, event in selected]


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_llm_prompt(
    history: List[dict],
    stable: StableUserProfile,
    candidate_changes: dict,
) -> str:
    profile_json = json.dumps(stable.to_dict(), indent=2)
    changes_json = json.dumps(candidate_changes, indent=2)
    selected = select_relevant_history_for_llm(history)

    skipped = [e for e in selected if e.get("event_type") == "skip"]
    repeated = [e for e in selected if e.get("event_type") == "repeat"]
    completed = [e for e in selected if e.get("event_type") == "complete"]

    def fmt(e: dict) -> str:
        title  = _sanitize(e.get("song_title", "?"))
        artist = _sanitize(e.get("song_artist", "?"))
        genre  = _sanitize(e.get("song_genre", "?"))
        mood   = _sanitize(e.get("song_mood", "?"))
        energy = e.get("song_energy", 0.0)
        ratio  = e.get("elapsed_ratio", 0.0)
        rcount = e.get("repeat_count", 0)
        etype  = e.get("event_type", "")
        if etype == "skip":
            return f"  - '{title}' by {artist} | {genre}/{mood} | energy={energy:.2f} | skipped at {ratio:.0%}"
        elif etype == "repeat":
            return f"  - '{title}' by {artist} | {genre}/{mood} | energy={energy:.2f} | repeated x{rcount}"
        else:
            return f"  - '{title}' by {artist} | {genre}/{mood} | energy={energy:.2f} | completed"

    sections = [
        f"Current profile:\n{profile_json}",
        f"\nCandidate changes from deterministic analysis:\n{changes_json}",
        f"\nSelected evidence window: {len(selected)} most recent interaction events (max {LLM_EVENT_LIMIT}).",
        "\nTreat skipped songs as negative evidence. Earlier skips should count as stronger dislike than late skips.",
        f"\nSKIPPED ({len(skipped)}):" + ("\n" + "\n".join(fmt(e) for e in skipped) if skipped else "\n  (none)"),
        f"\nREPEATED ({len(repeated)}):" + ("\n" + "\n".join(fmt(e) for e in repeated) if repeated else "\n  (none)"),
        f"\nCOMPLETED ({len(completed)}):" + ("\n" + "\n".join(fmt(e) for e in completed) if completed else "\n  (none)"),
        "\nRefine the candidate_changes JSON based on this evidence. Return only the JSON object.",
    ]

    prompt = "\n".join(sections)

    # Token budget: cap at ~3000 chars, trimming oldest events if needed
    if len(prompt) > 3000:
        # Rebuild with fewer selected events (keep most recent)
        for cutoff in range(len(selected) - 1, 0, -2):
            trimmed = selected[-cutoff:]
            skipped = [e for e in trimmed if e.get("event_type") == "skip"]
            repeated = [e for e in trimmed if e.get("event_type") == "repeat"]
            completed = [e for e in trimmed if e.get("event_type") == "complete"]
            sections[2] = f"\nSelected evidence window: {len(trimmed)} most recent interaction events (max {LLM_EVENT_LIMIT})."
            sections[3] = "\nTreat skipped songs as negative evidence. Earlier skips should count as stronger dislike than late skips."
            sections[4] = f"\nSKIPPED ({len(skipped)}):" + ("\n" + "\n".join(fmt(e) for e in skipped) if skipped else "\n  (none)")
            sections[5] = f"\nREPEATED ({len(repeated)}):" + ("\n" + "\n".join(fmt(e) for e in repeated) if repeated else "\n  (none)")
            sections[6] = f"\nCOMPLETED ({len(completed)}):" + ("\n" + "\n".join(fmt(e) for e in completed) if completed else "\n  (none)")
            prompt = "\n".join(sections)
            if len(prompt) <= 3000:
                break

    return prompt


# ── Post-parse guardrails ─────────────────────────────────────────────────────

def parse_and_guard(
    raw_json: str,
    stable: StableUserProfile,
    allow_genre_change: bool,
    allow_mood_change: bool,
) -> StableUserProfile:
    try:
        proposed = json.loads(raw_json)
    except json.JSONDecodeError:
        return stable

    if not isinstance(proposed, dict):
        return stable

    # Work on a mutable copy via asdict
    d = dataclasses.asdict(stable)

    for field, new_val in proposed.items():
        if field in ("favorite_artist", "scoring_mode", "version",
                     "last_updated", "update_reason", "previous_version"):
            continue  # protected — never accept from LLM

        if field == "favorite_genre":
            if allow_genre_change and isinstance(new_val, str) and new_val.strip():
                d["favorite_genre"] = new_val.strip().lower()
            continue

        if field == "favorite_mood":
            if allow_mood_change and isinstance(new_val, str) and new_val.strip():
                d["favorite_mood"] = new_val.strip().lower()
            continue

        if field == "likes_acoustic":
            if isinstance(new_val, bool):
                d["likes_acoustic"] = new_val
            continue

        if field == "preferred_mood_tags":
            if isinstance(new_val, list):
                d["preferred_mood_tags"] = [str(t) for t in new_val]
            continue

        if field == "preferred_decade":
            if isinstance(new_val, (int, float)):
                d["preferred_decade"] = int(new_val)
            continue

        if field == "target_tempo_bpm":
            if isinstance(new_val, (int, float)):
                new_tempo = int(new_val)
                old_tempo = int(d.get(field, 100))
                new_tempo = max(_TEMPO_MIN, min(_TEMPO_MAX, new_tempo))
                if abs(new_tempo - old_tempo) > _MAX_TEMPO_DELTA:
                    new_tempo = old_tempo + _MAX_TEMPO_DELTA * (1 if new_tempo > old_tempo else -1)
                d[field] = int(new_tempo)
            continue

        if field in _NUMERIC_FIELDS:
            try:
                new_f = float(new_val)
            except (TypeError, ValueError):
                continue
            old_f = float(d.get(field, 0.5))
            # Clamp range
            new_f = max(0.0, min(1.0, new_f))
            # Max delta guard
            if abs(new_f - old_f) > _MAX_DELTA:
                new_f = old_f + _MAX_DELTA * (1 if new_f > old_f else -1)
            d[field] = round(new_f, 3)

    try:
        return StableUserProfile(
            favorite_genre=d["favorite_genre"],
            favorite_mood=d["favorite_mood"],
            favorite_artist=d["favorite_artist"],
            scoring_mode=d["scoring_mode"],
            target_energy=d["target_energy"],
            target_danceability=d["target_danceability"],
            target_valence=d["target_valence"],
            target_live_energy=d["target_live_energy"],
            target_lyrical_depth=d["target_lyrical_depth"],
            target_instrumentalness=d["target_instrumentalness"],
            target_tempo_bpm=d["target_tempo_bpm"],
            desired_popularity=d["desired_popularity"],
            preferred_decade=d["preferred_decade"],
            likes_acoustic=d["likes_acoustic"],
            preferred_mood_tags=d["preferred_mood_tags"],
            version=d["version"],
            last_updated=d["last_updated"],
            update_reason=d["update_reason"],
            previous_version=d["previous_version"],
        )
    except (KeyError, TypeError):
        return stable


# ── LLM call ─────────────────────────────────────────────────────────────────

def _apply_changes(stable: StableUserProfile, changes: dict) -> StableUserProfile:
    """Apply candidate_changes dict directly to stable profile (deterministic path)."""
    raw = json.dumps(changes)
    return parse_and_guard(raw, stable, allow_genre_change=True, allow_mood_change=True)


def call_llm_reeval(
    history: List[dict],
    stable: StableUserProfile,
    changes: dict,
    allow_genre_change: bool,
    allow_mood_change: bool,
) -> Tuple[StableUserProfile, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n[LLM pass skipped: OPENAI_API_KEY not set — using deterministic update only]")
        candidate = _apply_changes(stable, changes)
        return candidate, "deterministic only (no API key)"

    try:
        from openai import OpenAI
    except ImportError:
        print("\n[LLM pass skipped: openai package not installed]")
        candidate = _apply_changes(stable, changes)
        return candidate, "deterministic only (openai not installed)"

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_llm_prompt(history, stable, changes)},
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        raw_json = response.choices[0].message.content.strip() or "{}"
    except Exception as exc:
        err = str(exc)
        if "API_KEY" in err or "credentials" in err.lower():
            print("\n[LLM pass skipped: invalid OPENAI_API_KEY]")
        candidate = _apply_changes(stable, changes)
        return candidate, "deterministic only (API error)"

    candidate = parse_and_guard(raw_json, stable, allow_genre_change, allow_mood_change)

    # If LLM vetoed all changes (returned {} or made no tracked modifications),
    # fall back to the deterministic candidate so valid evidence isn't silently discarded.
    _TRACKED = list(_NUMERIC_FIELDS) + [
        "target_tempo_bpm", "favorite_genre", "favorite_mood",
        "likes_acoustic", "preferred_mood_tags",
    ]
    llm_changed = any(getattr(candidate, f) != getattr(stable, f) for f in _TRACKED)
    if not llm_changed:
        candidate = _apply_changes(stable, changes)
        return candidate, "deterministic (LLM returned no changes)"

    return candidate, "hybrid (deterministic + OpenAI)"


# ── Session-end orchestrator ──────────────────────────────────────────────────

def update_profile_at_session_end(
    history_path: str,
    stable: StableUserProfile,
) -> Tuple[StableUserProfile, str]:
    history = load_last_n_history(history_path)

    if len(history) < _MIN_EVENTS:
        return stable, "no_change: insufficient history"

    changes, allow_genre, allow_mood = deterministic_update(history, stable)

    if not changes:
        return stable, "no_change: evidence below threshold"

    candidate, method = call_llm_reeval(history, stable, changes, allow_genre, allow_mood)

    # Build diff summary
    diff_parts = []
    for fname in list(_NUMERIC_FIELDS) + ["target_tempo_bpm", "favorite_genre", "favorite_mood",
                                           "likes_acoustic", "preferred_mood_tags"]:
        old_val = getattr(stable, fname)
        new_val = getattr(candidate, fname)
        if old_val != new_val:
            diff_parts.append(f"{fname}: {old_val!r}→{new_val!r}")

    if not diff_parts:
        return stable, "no_change: evidence below threshold"

    shift_summary, ranking_changed = summarize_recommendation_shift(stable, candidate)
    major_field_changed = any(
        getattr(stable, field) != getattr(candidate, field)
        for field in ("favorite_genre", "favorite_mood", "preferred_mood_tags", "target_tempo_bpm")
    )
    if not ranking_changed and not major_field_changed:
        return stable, f"no_change: low recommendation impact ({shift_summary})"

    reason = f"{method} — " + ", ".join(diff_parts[:4])
    if len(diff_parts) > 4:
        reason += f" (+{len(diff_parts) - 4} more)"
    reason += f" | {shift_summary}"

    # Versioning
    new_stable = StableUserProfile(
        favorite_genre=candidate.favorite_genre,
        favorite_mood=candidate.favorite_mood,
        favorite_artist=stable.favorite_artist,        # always preserved
        scoring_mode=stable.scoring_mode,              # always preserved
        target_energy=candidate.target_energy,
        target_danceability=candidate.target_danceability,
        target_valence=candidate.target_valence,
        target_live_energy=candidate.target_live_energy,
        target_lyrical_depth=candidate.target_lyrical_depth,
        target_instrumentalness=candidate.target_instrumentalness,
        target_tempo_bpm=candidate.target_tempo_bpm,
        desired_popularity=candidate.desired_popularity,
        preferred_decade=candidate.preferred_decade,
        likes_acoustic=candidate.likes_acoustic,
        preferred_mood_tags=candidate.preferred_mood_tags,
        version=stable.version + 1,
        last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        update_reason=reason,
        previous_version=stable.to_dict(),
    )

    return new_stable, reason
