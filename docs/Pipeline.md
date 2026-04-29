# Music Recommender — V2 Pipeline Architecture

> **This document describes the target v2 design.**
> The current codebase is v1: stateless scoring, flat `UserProfile`, ASCII table output.
> This architecture is aspirational relative to the current repo state.

---

## Overview

The v2 system runs two interlocking loops:

1. **In-session adaptation loop** — steers recommendations in real time based on behavioral signals during playback
2. **End-of-session profile learning loop** — updates the long-term user profile once at session end using a hybrid deterministic + LLM pipeline

---

## State Types

| Type | Scope | Persisted |
|---|---|---|
| `StableUserProfile` | Cross-session long-term taste | Yes — `data/user_profile.json` |
| `SessionState` | Current session only, ephemeral nudges | No |
| `InteractionEvent` | Immutable per-song interaction record | Yes — `data/history.jsonl` (rolling 75) |

### StableUserProfile
Long-term taste defaults. Loaded at session start; written at session end only if evidence thresholds are met. Carries version metadata and a shallow `previous_version` snapshot for rollback context.

### SessionState
Holds ephemeral numeric nudges (`energy_nudge`, `danceability_nudge`, `valence_nudge`) applied on top of `StableUserProfile` during the current session. Dies when the session ends — never persisted.

### InteractionEvent
Immutable record of one interaction: `complete`, `skip`, `repeat`, or `quit`. `quit` signals session termination only and is explicitly **not** treated as a dislike signal.

---

## Loop 1: In-Session Adaptation

Runs continuously during playback. Responds to strong behavioral signals by recomputing the recommendation queue using a merged preference view.

```
[StableUserProfile] ──┐
                       ├──▶ merge_prefs() ──▶ recommend_songs() ──▶ queue
[SessionState nudges] ─┘         ▲
                                  │
                   apply_session_nudge() ◀── skip / repeat events
```

### Recompute Triggers

| Signal | Condition | Action |
|---|---|---|
| Rapid skips | `skips_since_last_recompute >= 2` | Recompute queue with current nudged prefs |
| Repeat | Any repeat event | Recompute queue + reset song timer |
| Queue exhaustion | `len(queue) == 0` | Recompute queue |
| Quit | `q` or Ctrl+C | Session termination — no recompute, go to Loop 2 |

### Session Nudges

After each skip, `apply_session_nudge()` nudges `energy_nudge` and `valence_nudge` **away** from the skipped song's attributes (magnitude `0.05`, capped at `±0.25`). After a repeat, nudges move **toward** the song's attributes. These nudges only affect the current session's `merge_prefs()` output — the `StableUserProfile` is never mutated during playback.

Already-played song IDs (`session.played_ids`) are excluded from all recompute calls.

---

## Loop 2: End-of-Session Profile Learning

Runs once when the user quits. Updates `StableUserProfile` using a hybrid deterministic + LLM pipeline.

```
[InteractionEvent log] ──▶ deterministic_update() ──▶ candidate_changes
                                                              │
                                                              ▼
                                                    call_llm_reeval()
                                                    (advisory, bounded)
                                                              │
                                                              ▼
                                                    parse_and_guard()
                                                    (clamp · max delta ±0.3
                                                     genre/mood threshold gate
                                                     enum validation
                                                     protected fields)
                                                              │
                                               ┌──── no_change? ────┐
                                               │                    │
                                    keep prior profile      save new StableUserProfile
                                    (do not write file)     (versioned + reason summary)
```

### Step 1 — Minimum history gate
If fewer than 3 events in history → `no_change`, skip everything.

### Step 2 — Deterministic aggregation
Computes candidate field changes purely from statistics:
- **Numeric fields**: mean energy of completed+repeated songs vs current target; nudge by `0.15` if gap > `0.1`
- **Preferred mood tags**: union of repeated-song moods, minus moods that appear only in early skips (`elapsed_ratio < 0.3`)
- **Genre/mood change**: allowed only if `≥50%` of skips involve the current genre AND `≥50%` of completions involve a different genre (relative ratios, not absolute fractions — avoids false triggers with 80 songs across 16 genres)

Returns a `candidate_changes` dict (field → new_value). Fields with insufficient evidence are omitted.

### Step 3 — LLM advisory pass
Sends the current profile, `candidate_changes`, and selected sanitized history events to `gpt-4.1-mini`. The LLM may **refine or remove** proposed changes but may **not invent** new ones or touch protected fields. If `OPENAI_API_KEY` is not set, this step is skipped and deterministic changes are applied directly.

### Step 4 — Post-parse guardrails
All LLM output is validated before acceptance:

| Check | Rule |
|---|---|
| Float clamping | All float fields → `[0.0, 1.0]` |
| Max delta | `abs(new - old) ≤ 0.3` per numeric field; clip to boundary |
| Genre/mood gate | Code re-checks the ratio threshold; overrides if not met |
| Enum validation | `scoring_mode` must be one of 4 valid values |
| Protected fields | `favorite_artist`, `scoring_mode`, `version`, `last_updated`, `update_reason`, `previous_version` — never accepted from raw LLM output. `favorite_artist` may still be set by deterministic evidence. |

### Step 5 — No-change check
If no fields differ after guardrails → `no_change` is returned. `user_profile.json` is **not** written. The prior profile is preserved intact.

### Step 6 — Versioning
Every saved profile carries:
- `version`: incremented by 1
- `last_updated`: ISO 8601 UTC timestamp
- `update_reason`: human-readable summary of changed fields and update method
- `previous_version`: shallow snapshot of the prior profile for rollback context

---

## Input Guardrails (before sending to LLM)

- All string fields from history events are sanitized: newlines, carriage returns, and null bytes stripped; truncated to 80 chars
- Prompt is token-budgeted: capped at ~3000 chars by dropping oldest events until it fits

---

## File Layout

```
src/
  agentic_flow.py  — startup workflow orchestration (Spotify sync → profile update → versioning)
  models.py        — StableUserProfile, SessionState, InteractionEvent dataclasses
  player.py        — NowPlayingUI, KeyboardListener, merge_prefs, playback loop
  llm_reeval.py    — history I/O, deterministic_update, LLM call, parse_and_guard
  recommender.py   — unchanged scoring engine (recommend_songs, score_song)
  main.py          — CLI entrypoint; delegates startup flow to agentic_flow.py

data/
  songs.csv            — 80-song catalog
  history.jsonl        — rolling 75-event interaction log (append-only, auto-trimmed)
  user_profile.json    — latest saved StableUserProfile (written at session end)

docs/
  Pipeline.md      — this document
```

---

## Running

```bash
# Install dependency
pip install openai

# Set API key (optional — deterministic-only update if absent)
export OPENAI_API_KEY=sk-...

# Now Playing mode
python -m src.main --now-playing

# Original table output (unchanged)
python -m src.main
```

### Keyboard controls

| Key | Action |
|---|---|
| `s` or `→` | Skip to next song |
| `r` | Repeat current song (resets timer, recomputes queue) |
| `q` | Quit and save updated profile |
| `Ctrl+C` | Emergency quit (also saves history and profile) |
