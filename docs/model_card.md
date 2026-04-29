# Model Card: VibeFinder

**Version:** 2.0 (Spotify + LLM integration)  
**Date:** 2026-04-29  
**Author:** Yueh Chung Wang  

---

## 1. Model Overview

VibeFinder is a personal music recommender that combines content-based filtering, real Spotify listening data, and two OpenAI language models to produce and continuously refine song recommendations.

The system has two distinct AI components:

| Component | Model | Role |
| --- | --- | --- |
| Audio Feature Estimator | `gpt-4o-mini` | Estimates energy, valence, danceability, tempo, mood, and 7 other audio features from track name + artist |
| Profile Re-evaluator | `gpt-4.1-mini` | Refines deterministic candidate profile changes using listening evidence (skip / complete / repeat events) |

Neither model is fine-tuned. Both are used via the OpenAI Chat Completions API with structured prompts. The re-evaluator uses few-shot examples in its system prompt to constrain behavior.

---

## 2. Intended Use

**Primary use:** Personal music recommendation for a single Spotify user. The system adapts its recommendations over time based on the user's actual listening behavior.

**Appropriate contexts:**

- Individual use connected to your own Spotify account
- Offline use with a pre-populated `songs.csv` library
- Educational exploration of content-based filtering and LLM-assisted profile management

**Out-of-scope uses:**

- Multi-user recommendation (profiles represent one person; shared accounts will blend tastes incoherently)
- Commercial music discovery or playlist generation for third parties
- Real-time streaming or playlist injection into Spotify

---

## 3. System Architecture

The system has four layers that run in sequence at every startup:

```text
Spotify API  →  Sync Layer  →  Core Engine  →  Terminal Player
(3 sources)     (skip infer     (13-dim score    (s/r/q events
                + OpenAI est)    + LLM profile)    → history.jsonl)
```

**Sync Layer** (`spotify_sync.py`)  
Fetches recently played (50 tracks), top tracks (150 across short/medium/long term), and liked songs (50 tracks) via Spotify OAuth. New tracks not already in the library are sent to `gpt-4o-mini` for audio feature estimation. Skip vs. complete is inferred by dividing the gap between consecutive `played_at` timestamps by `duration_ms`.

**Core Engine** (`recommender.py` + `llm_reeval.py`)  
Scores every song in the library across 13 weighted dimensions, then applies a diversity re-ranker that penalizes back-to-back same-artist or same-genre picks. After sync, `llm_reeval.py` runs a deterministic analysis of the history to produce bounded candidate changes, then passes them to `gpt-4.1-mini` for refinement before committing a new profile version.

**Data Store**  
Three files drive all state: `songs.csv` (song library, 228+ tracks), `history.jsonl` (every interaction event, capped at 75), and `user_profile.json` (versioned taste profile). A fourth file, `data/.spotify_imported.json`, prevents top tracks and liked songs from being double-counted across sync runs.

---

## 4. Input Data

### Song Library (`songs.csv`)

| Field | Type | Source |
| --- | --- | --- |
| `title`, `artist` | string | Spotify metadata |
| `genre` | string (16 values) | OpenAI `gpt-4o-mini` estimate or search category |
| `mood` | string (17 values) | OpenAI `gpt-4o-mini` estimate |
| `energy`, `valence`, `danceability`, `acousticness`, `instrumentalness` | float [0, 1] | OpenAI `gpt-4o-mini` estimate |
| `live_energy`, `lyrical_depth` | float [0, 1] | OpenAI `gpt-4o-mini` estimate |
| `tempo_bpm` | int [40, 220] | OpenAI `gpt-4o-mini` estimate |
| `popularity` | int [0, 100] | Spotify metadata (real) |
| `release_decade` | int | Derived from Spotify `release_date` |
| `mood_tags` | pipe-separated string | OpenAI `gpt-4o-mini` estimate |

**Note:** Spotify's `/audio-features` endpoint is restricted for apps created after November 2024. All audio features except `popularity` and `release_date` are OpenAI estimates, not Spotify measurements.

### Interaction History (`history.jsonl`)

Each line is a JSON object recording one interaction event:

| Field | Values |
| --- | --- |
| `event_type` | `complete`, `skip`, `repeat`, `quit` |
| `elapsed_ratio` | float [0, 1] — fraction of track heard |
| `song_energy`, `song_tempo_bpm` | carried from `songs.csv` at time of event |
| `session_id` | `spotify-recent`, `spotify-top-{term}`, `spotify-liked`, or in-app session ID |

Events from Spotify are inferred (skip / complete from timestamp gaps; repeat for liked songs). Events from the terminal player are captured in real time.

### User Profile (`user_profile.json`)

The profile is a versioned JSON object with 14 fields. Key fields:

| Field | Type | Protected? |
| --- | --- | --- |
| `favorite_genre` | string | No — LLM-adjustable with evidence gate |
| `favorite_mood` | string | No — LLM-adjustable with evidence gate |
| `favorite_artist` | string | **Yes — never modified by LLM** |
| `scoring_mode` | string | **Yes — never modified by LLM** |
| `target_energy`, `target_tempo_bpm`, etc. | float / int | No — bounded ±0.3 per session |
| `version` | int | **Yes — managed by versioning system only** |

---

## 5. Scoring Algorithm

### 13-Dimensional Weighted Score

Each candidate song is scored as a weighted sum across 13 dimensions:

| Dimension | `balanced` weight | Score method |
| --- | --- | --- |
| Genre match | 0.16 | Exact = 1.0; similarity map (e.g. pop↔indie pop = 0.8); else 0.0 |
| Mood match | 0.14 | Exact = 1.0; similarity map (e.g. chill↔relaxed = 0.7); else 0.0 |
| Energy | 0.12 | `1 - abs(song - user)` |
| Danceability | 0.10 | `1 - abs(song - user)` |
| Mood tags | 0.08 | Jaccard overlap of tag sets |
| Popularity | 0.08 | `1 - abs(song/100 - user)` |
| Acousticness | 0.07 | `song_acousticness` if `likes_acoustic`, else `1 - song_acousticness` |
| Tempo | 0.06 | `1 - abs(song - user) / 100` |
| Valence | 0.06 | `1 - abs(song - user)` |
| Release decade | 0.05 | `1 - abs(song - user) / 40` |
| Live energy | 0.04 | `1 - abs(song - user)` |
| Lyrical depth | 0.04 | `1 - abs(song - user)` |
| Instrumentalness | 0.03 | `1 - abs(song - user)` |

If `favorite_artist` is set, an artist bonus weight of 0.08 is added (redistributed from `popularity` and `instrumentalness`).

### Diversity Re-ranker

After initial scoring, a greedy re-ranker penalizes redundancy in the output queue:

- Same artist as any already-selected song: **−0.08**
- Same genre as any already-selected song: **−0.05**

### Scoring Modes

Four preset weight configurations are available:

| Mode | Emphasis |
| --- | --- |
| `balanced` | Equal spread across all dimensions (default) |
| `genre_first` | Genre weight 0.26; decade 0.09 |
| `mood_first` | Mood 0.24; mood tags 0.16 |
| `energy_focused` | Energy 0.26; tempo 0.10; live energy 0.08 |

---

## 6. Profile Update Algorithm

### Step 1 — Deterministic Analysis

Before the LLM is called, a rule-based pass computes bounded candidate changes:

- **Energy:** if mean energy of completed songs differs from profile target by > 0.05, nudge target by ±0.15
- **Tempo:** if mean tempo of completed songs differs by > 10 BPM, nudge by ±20 BPM
- **Mood tags:** collect moods from all `repeat` events, subtract moods of early-skip songs (ratio < 0.3)
- **Genre/mood change gate:** only opens if the dominant completed genre/mood appears in ≥35% of positive events AND differs from current profile AND has ≥3 skip events on the current value

### Step 2 — LLM Refinement

The candidate changes are passed to `gpt-4.1-mini` with:

- The current profile as JSON
- Up to 20 selected interaction events (prioritizing early skips and repeats)
- Four few-shot examples in the system prompt showing correct vs incorrect refinement behavior
- A strict instruction: "refine the candidates — do not invent new fields; never emit protected fields"

The LLM may reduce a proposed delta, veto a genre/mood change if evidence looks weak, or return `{}` to discard all candidates.

### Step 3 — Guardrails

After the LLM response is parsed, `parse_and_guard()` enforces:

- Protected fields (`favorite_artist`, `scoring_mode`, `version`, `last_updated`, `update_reason`, `previous_version`) are silently dropped
- All float fields clamped to [0.0, 1.0]
- Max delta per session: ±0.3 for float fields, ±30 for tempo
- Tempo hard bounds: [40, 220] BPM
- If LLM returns no changes, the deterministic candidate is applied instead (no silent discard)

### Step 4 — Impact Gate

A profile update is only saved if it changes the top-3 recommendation list **or** modifies a major field (`favorite_genre`, `favorite_mood`, `preferred_mood_tags`, `target_tempo_bpm`). Minor numeric nudges that don't affect visible output are discarded.

---

## 7. Performance and Evaluation

### Unit Tests — 13 / 13 passed

`pytest tests/` covers scoring functions, genre/mood similarity maps, LLM fallback behavior, prompt token budgeting, and evidence window selection.

### Synthetic Test Harness — 6 / 6 scenarios, 130+ checks passed

`python -m src.evaluate_llm_reliability` runs 5 predefined behavioral scenarios plus a live-data scenario:

| Scenario | Behavioral guarantee tested | Result |
| --- | --- | --- |
| 1 — High-energy listener | Energy + tempo nudge upward | PASS 100% |
| 2 — Heavy genre-skipper | Genre-change gate opens correctly | PASS 100% |
| 3 — Protected field injection | Guardrail blocks malicious JSON | PASS 100% |
| 4 — Insufficient history | No update below 5 events | PASS 100% |
| 5 — Repeat signal | Mood tags updated from repeat events | PASS 100% |
| 6 — Live data + LLM | All guardrails on real history | PASS 100% |

### LLM Reliability — 20 / 20 guardrail checks on 75 live events, 15 profile versions

All guardrails held across every historical profile version: no protected field was ever modified, no numeric field exceeded ±0.3 per update, and all tempo values stayed within [40, 220] BPM.

---

## 8. Limitations and Known Failure Modes

**Audio feature estimation error**  
`gpt-4o-mini` estimates energy, valence, danceability, and other features from track name and artist alone — without audio signal. Two very different songs with similar-sounding titles can receive similar feature vectors. Estimates are consistent enough for relative ranking but should not be treated as ground truth.

**Skip inference is probabilistic**  
Skip detection uses the timestamp gap between consecutive `played_at` entries divided by `duration_ms`. This correctly identifies most skips but can misclassify a short song (<90 seconds) that was fully heard as a skip if the next track started quickly. It also cannot detect pause-resume sessions.

**Recency bias in top tracks**  
Spotify's short-term top tracks (last 4 weeks) dominate early syncs. All-time favorites that haven't been played recently are underrepresented until long-term top tracks and liked songs accumulate in the history.

**Single-user design**  
The profile represents one listener. Shared Spotify accounts produce a blended signal that the system has no mechanism to disentangle.

**Genre taxonomy gap**  
Spotify's rich genre labels (bedroom pop, dark trap, hyperpop) are mapped into 16 internal buckets. Sub-genre nuance is lost, which can cause a "dark trap" track to score as "hip-hop" against a "pop" profile — lower than it should rank.

**LLM genre/mood veto can be over-conservative**  
The few-shot examples train the LLM to veto weak genre changes. In edge cases where the user is genuinely transitioning tastes (e.g., 40% hip-hop completes, 35% pop), the LLM may require multiple sessions before accepting the change.

**No cold-start recovery**  
The system defaults to `pop/happy` with neutral numeric targets when no profile or history exists. The first sync can dramatically shift the profile (v1 → v14 in a single session), which can feel jarring if the defaults were far from the user's actual taste.

---

## 9. Ethical Considerations

**Data scope**  
The system processes only the authenticated user's own Spotify data. No data is sent to third parties other than the OpenAI API for feature estimation and profile refinement. All state is stored locally in three plain-text files.

**Echo chamber risk**  
A pure content-based recommender that converges on a stable profile can create a feedback loop: recommendations narrow → user completes them → profile reinforces the narrow range. The ±0.3 max-delta guard, the LLM veto, and the diversity re-ranker reduce this risk but do not eliminate it. A user who only ever plays hip-hop/sad will eventually see almost exclusively hip-hop/sad recommendations.

**Transparency**  
Every profile update includes a `update_reason` field explaining what changed and why (e.g., `"hybrid (deterministic + OpenAI) — target_energy: 0.6→0.45 | top3 changed: ..."`). Users can run `--sync-profile-preview` before committing any update.

**No commercial intent**  
The system makes no purchasing recommendations and does not transmit listening data to advertisers or third parties.

---

## 10. Dependencies and Versions

| Package | Purpose |
| --- | --- |
| `spotipy` | Spotify Web API OAuth client |
| `openai` | `gpt-4o-mini` (feature estimation) and `gpt-4.1-mini` (profile re-eval) |
| `python-dotenv` | Credential loading from `.env` |
| `pytest` | Unit test runner |

Python 3.10+ required.

---

## 11. Version History

| Version | Changes |
| --- | --- |
| 1.0 | Content-based recommender with 20 manual songs, 7 scoring dimensions, no external APIs |
| 2.0 | Spotify OAuth sync (3 sources), OpenAI audio feature estimation, LLM profile re-evaluator, 13-dimensional scoring, diversity re-ranker, versioned profile with guardrails, synthetic test harness, few-shot specialization |
