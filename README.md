# AI-Assisted Music Recommender

A personalized music recommendation system that learns your taste from real Spotify listening data and adapts its recommendations over time using an LLM-driven profile.

**Video walkthrough:** [Loom link — add after recording]  
**GitHub:** [your-repo-url]

---

## Original Project (Modules 1–3)

**VibeFinder** — a content-based music recommender built from scratch.
The system represented songs and user taste as structured data, then computed weighted similarity scores across features like genre, mood, energy, tempo, and danceability to produce ranked recommendations. It was evaluated against edge cases (contradictory profiles, out-of-range inputs, unknown words) to expose how feature weighting shapes — and can distort — output quality.

---

## What This Project Does (Module 4+)

VibeFinder now connects to your real Spotify account. Every time you launch the terminal player it:

1. Pulls your **recently played**, **top tracks** (short/medium/long term), and **liked songs** from Spotify
2. Adds any new tracks to the song library with OpenAI-estimated audio features
3. Infers whether you **skipped or completed** each Spotify play from timestamp gaps vs. track duration
4. Writes those play events to a history log and runs an **LLM re-evaluation** of your taste profile
5. Launches an interactive terminal player that recommends songs from your updated library

The result is a recommender that improves with every listening session — on Spotify or in the app — without you ever filling out a form.

---

## System Architecture

flowchart TD
    subgraph Spotify["Spotify API OAuth"]
        SP1["Recently Played<br/>50 tracks + timestamps"]
        SP2["Top Tracks<br/>short / medium / long term"]
        SP3["Liked Songs<br/>hearted tracks"]
    end

    subgraph Sync["spotify_sync.py — Sync Layer"]
        SKIP["Skip Inference<br/>timestamp gap ÷ duration_ms"]
        NEW["New Track Detection<br/>title + artist dedup"]
    end

    subgraph AI["AI Layer"]
        OAI["OpenAI gpt-4o-mini<br/>Estimate audio features<br/>genre · mood · energy · tempo"]
        LLM["OpenAI gpt-4.1-mini<br/>LLM Profile Re-evaluator<br/>refines candidate changes"]
    end

    subgraph Store["Data Store"]
        CSV[("songs.csv<br/>216 tracks")]
        HIST[("history.jsonl<br/>skip · complete · repeat events")]
        PROF[("user_profile.json<br/>versioned taste profile")]
        REG[(".spotify_imported.json<br/>import dedup registry")]
    end

    subgraph Core["Core Engine"]
        REC["recommender.py<br/>Weighted scoring<br/>13 audio dimensions"]
        REEVAL["llm_reeval.py<br/>Deterministic analysis<br/>+ LLM refinement"]
    end

    subgraph UI["Terminal Player"]
        PLAYER["player.py<br/>Now Playing UI<br/>s = skip · r = repeat · q = quit"]
        HUMAN(["Human Listener"])
    end

    SP1 --> SKIP
    SP2 --> NEW
    SP3 --> NEW
    SKIP --> HIST
    NEW --> OAI --> CSV
    CSV --> REC
    HIST --> REEVAL
    REEVAL --> LLM --> PROF
    PROF --> REC
    REC --> PLAYER
    PLAYER <--> HUMAN
    HUMAN -- "skip / repeat / complete" --> HIST
    REG --> NEW
    NEW --> REG
---

## Architecture Overview

The system has four layers:

**Sync Layer** (`spotify_sync.py`) — Connects to Spotify via OAuth at startup. It fetches three sources (recently played, top tracks, liked songs), deduplicates against the existing library, and sends new tracks to OpenAI for feature estimation. Skip vs. complete is inferred by dividing the timestamp gap between consecutive plays by the track's `duration_ms` — if the ratio is below 0.5, the play is recorded as a skip.

**Data Store** — Three files drive all state: `songs.csv` (the song library), `history.jsonl` (every interaction event), and `user_profile.json` (the evolving taste profile). A fourth file, `.spotify_imported.json`, tracks which Spotify track IDs have already been imported so top tracks and liked songs are never double-counted.

**Core Engine** — `recommender.py` scores every candidate song across 13 weighted dimensions (genre, mood, energy, tempo, danceability, valence, acousticness, popularity, release decade, mood tags, live energy, lyrical depth, instrumentalness) with a diversity re-ranker that penalizes back-to-back same-artist or same-genre picks. `llm_reeval.py` runs after each session: it first computes deterministic candidate changes from listening statistics, then passes those candidates to GPT-4.1-mini to refine or veto them.

**Human in the Loop** — The terminal player captures every `skip`, `repeat`, and `complete` event in real time. Users can also run `--sync-profile-preview` to review what the LLM wants to change before committing.

---

## Setup

### Prerequisites

- Python 3.10+
- A [Spotify Developer app](https://developer.spotify.com/dashboard) with `http://127.0.0.1:8888/callback` as a Redirect URI
- An OpenAI API key

### Install

```bash
git clone <your-repo-url>
cd ai_assisted_music_recommender
pip install -r requirements.txt
```

### Configure `.env`

```env
OPENAI_API_KEY=sk-...
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

### Run

```bash
# Launch the terminal player (auto-syncs Spotify on startup)
python src/main.py --now-playing

# Skip Spotify sync (offline / faster startup)
python src/main.py --now-playing --no-sync

# See what the LLM wants to change before saving
python src/main.py --sync-profile-preview

# Apply profile update
python src/main.py --sync-profile

# Populate library from Spotify genre search (no login needed)
python src/spotify_fetch.py --genres pop rock jazz --limit 20

# Run tests
pytest

# Run the reliability evaluation harness
python -m src.evaluate_llm_reliability
```

> **First run note:** `--now-playing` (with sync) opens your browser for Spotify login. After you approve access, the terminal returns automatically and sync begins. Use `--no-sync` to skip this if you want to start offline.

---

## Sample Interactions

### 1 — Terminal Player Session

```text
════════════════════════════════════════════════════════════════
  NOW PLAYING                              [queue #0 | v14]
════════════════════════════════════════════════════════════════
  Title   : Hurt (feat. Jessica Parry)
  Artist  : X-Boxin
  Genre   : hip-hop | Mood: sad | BPM: 90
  Energy  : 0.50 | Valence: 0.40 | Score: 0.77
════════════════════════════════════════════════════════════════
  s / RIGHT Skip    r Repeat    q Quit
════════════════════════════════════════════════════════════════
  [==================>  ]  18s / 30s
```

### 2 — Spotify Sync Output (startup)

```text
[startup] Syncing Spotify listening history...
[sync] Recently played : 50 tracks
[sync] Top tracks       : 150 (50/50/50 short/med/long)
[sync] Liked songs      : 50 tracks
[sync] New tracks       : 167 — estimating features...
[sync] songs.csv        : 216 total tracks (+167)
[sync] History events   : 8 recent (8 complete / 0 skip) + 131 top + 50 liked
[startup] Profile updated to v14: deterministic — preferred_mood_tags: []→['chill']
```

### 3 — Profile Sync Preview

```text
════════════════════════════════════════════════════════════════════
PROFILE SYNC PREVIEW
════════════════════════════════════════════════════════════════════
version: 13 -> 14
genre  : hip-hop -> hip-hop
mood   : sad -> sad
energy : 0.75 -> 0.6
tempo  : 140 -> 120
tags   : [] -> ['chill']
reason : deterministic — target_energy: 0.75→0.6, target_tempo_bpm: 140→120
top5 before: ['Hurt (feat. Jessica Parry)', 'Don't Lose Sleep', ...]
top5 after : ['Hurt (feat. Jessica Parry)', 'Last December', ...]
```

### 4 — Top 5 Recommendations (CLI mode)

```text
════════════════════════════════════════════════════════════════════════════
TOP RECOMMENDATIONS
mode=balanced | genre=hip-hop | mood=sad | energy=0.6 | tempo=120
════════════════════════════════════════════════════════════════════════════
+--+--------------------+----------------+--------+------+------+
| # | Title             | Artist         | Genre  | Mood | Score|
+--+--------------------+----------------+--------+------+------+
| 1 | Hurt (feat. ...)  | X-Boxin        | hip-hop| sad  | 0.77 |
| 2 | Last December     | ...            | hip-hop| sad  | 0.74 |
| 3 | Don't Lose Sleep  | X-Boxin        | hip-hop| sad  | 0.71 |
+--+--------------------+----------------+--------+------+------+
```

---

## Design Decisions

**Why content-based filtering over collaborative filtering**
The system only has one user's data. Collaborative filtering needs many users to find neighbors — without that, it degenerates to popularity bias. Content-based scoring is transparent, debuggable, and works from session one.

**Why OpenAI to estimate audio features**
Spotify restricted its `/audio-features` endpoint to apps created before November 2024. Rather than abandon real track data, using `gpt-4o-mini` to estimate energy, valence, danceability, etc. from track name and artist costs roughly $0.002 per 50 songs and produces estimates consistent enough for relative ranking.

**Why a separate deterministic pass before the LLM**
Letting the LLM change the profile freely risks arbitrary drift. The deterministic layer (`llm_reeval.py`) computes bounded candidate changes from actual statistics (mean energy of completed songs, genre frequency in skips vs. completes). The LLM only refines those candidates — it cannot invent new fields or override protected ones like `favorite_artist` or `scoring_mode`.

**Why three Spotify sources instead of just recently played**
Recently played skews toward the last few hours. Top tracks (short/medium/long term) give a more stable signal. Liked songs are written as `repeat` events — the strongest positive signal — because hearting a song is a deliberate act.

**Trade-off: skip inference is probabilistic**
Skip detection compares the gap between consecutive `played_at` timestamps to `duration_ms`. This works well but can misclassify a short song listened to fully as a skip if the next track started quickly. A future improvement would poll `/me/player` in real time.

---

## Reliability and Evaluation

### Integrated Guardrails (always active in production)

The primary reliability mechanism is a guardrail layer built directly into the main application loop. Every LLM response passes through `parse_and_guard()` in `llm_reeval.py` before any change is applied to the profile. This runs on every sync and every in-app session — not as a test, but as part of the normal execution path.

| Guardrail | Where it runs | What it enforces |
| --- | --- | --- |
| Protected field block | `parse_and_guard()` | `favorite_artist`, `scoring_mode`, `version` are silently dropped from any LLM response — even if the LLM explicitly tries to modify them |
| Numeric range clamp | `parse_and_guard()` | All float fields kept within `[0.0, 1.0]`; tempo kept within `[40, 220]` BPM |
| Max-delta cap | `parse_and_guard()` | No float field moves more than `±0.3` per session; tempo moves at most `±30` BPM |
| LLM fallback | `call_llm_reeval()` | If OpenAI is unavailable or returns an error, the deterministic candidate is applied instead — the profile still improves |
| Impact gate | `update_profile_at_session_end()` | The updated profile is only saved if it changes the top-3 recommendation list or modifies a major field — minor noise is discarded |
| Minimum evidence gate | `update_profile_at_session_end()` | No update fires with fewer than 5 interaction events |

**Why this counts as a reliability feature, not just error handling:** these guardrails change the system's behavior on every run. A LLM response that passes `json.loads` but violates a constraint is silently corrected rather than accepted or rejected outright. The system degrades gracefully under bad LLM output instead of crashing or drifting.

---

### Automated Tests — 13 / 13 passed

`pytest` covers the scoring functions, genre/mood similarity maps, LLM fallback behavior, prompt token budgeting, and evidence window selection.

| Test | What it checks |
| --- | --- |
| `test_recommend_returns_songs_sorted_by_score` | Top result matches user's genre + mood |
| `test_for_genre_similarity` | `pop→indie pop = 0.8`, `pop→classical = 0.0` |
| `test_for_mood_similarity` | `happy→joyful = 0.8`, `happy→angry = 0.0` |
| `test_score_song_uses_tempo_preference` | Closer BPM scores higher |
| `test_explain_recommendation_returns_non_empty_string` | Explanation is always generated |
| `test_recommender_accepts_csv_path` | Loads real `songs.csv` correctly |
| `test_call_llm_reeval_without_api_key_falls_back_to_deterministic` | Graceful degradation |
| `test_call_llm_reeval_on_api_error_falls_back_to_deterministic` | Graceful degradation |
| `test_call_llm_reeval_uses_llm_json_response_when_available` | LLM output is applied correctly |
| `test_build_llm_prompt_limits_selected_events_and_mentions_skip_signal` | Prompt stays within token budget |
| `test_select_relevant_history_for_llm_prioritizes_early_skips` | Evidence window selection |
| `test_summarize_recommendation_shift_reports_when_top_recommendations_change` | Profile diff is meaningful |

---

### Evaluation Harness — 6 / 6 scenarios, 130+ checks passed

`python -m src.evaluate_llm_reliability` verifies the integrated guardrails behave correctly across predefined scenarios. Each scenario builds a synthetic history, runs the full update pipeline, and checks both the guardrail contracts and the expected behavioral outcome.

| Scenario | What it verifies | Result |
| --- | --- | --- |
| 1 — High-energy listener | Energy + tempo nudge upward when all completes are high-energy | PASS 100% |
| 2 — Heavy genre-skipper | Genre-change gate opens after ≥35% skips on current genre | PASS 100% |
| 3 — Protected field injection | `favorite_artist`, `scoring_mode`, `version` survive a malicious LLM JSON | PASS 100% |
| 4 — Insufficient history | No update fires below the minimum event threshold | PASS 100% |
| 5 — Repeat signal | `preferred_mood_tags` updated after 8 repeat events on "chill" songs | PASS 100% |
| 6 — Live data (+ LLM) | All guardrails hold on real `history.jsonl` with an actual OpenAI call | PASS 100% |

---

### Error Handling

- Spotify sync fails gracefully: missing credentials or a network failure leaves `songs.csv` unchanged and starts the player with the existing library
- LLM API errors and missing `OPENAI_API_KEY` both fall back to the deterministic update — the profile still improves without OpenAI
- Malformed JSON from the LLM is caught by `json.loads`; if parsing fails entirely, the current profile is returned unchanged

---

## Reflection and Ethics

### Limitations and Biases

- **Recency bias** — Recently played and top tracks (short-term) dominate the history log early on. Songs listened to years ago but not recently are underrepresented.
- **OpenAI estimation error** — Audio feature estimates are approximations. Two very different songs with similar names could receive similar feature vectors.
- **Skip inference accuracy** — Short songs (under 90 seconds) may be falsely flagged as skips even when fully heard.
- **Single-user design** — The profile represents one person's taste. If multiple people use the same Spotify account, the recommendations will blend their preferences incoherently.
- **Genre taxonomy gap** — The system maps Spotify's broad genres into 16 internal buckets. Sub-genres like "bedroom pop" or "dark trap" lose nuance in the mapping.

### Could This Be Misused?

The system only processes your own Spotify data and stores everything locally. It does not share data with third parties or make purchasing recommendations. The main risk is **over-personalization**: if the LLM profile updates too aggressively, recommendations can become an echo chamber — always the same mood and genre. The `±0.3` max-delta guard and the LLM veto layer are specifically designed to prevent this.

### What Surprised Me During Testing

The most surprising finding was how quickly the profile adapted to Liked Songs. After the first sync, 50 liked tracks were imported as `repeat` events. The LLM immediately detected that my actual listening leans toward **hip-hop/sad** rather than the initial **pop/happy** default, and the profile evolved through 14 versions in a single day — all without me explicitly rating anything.

### Collaboration With AI

**Helpful suggestion:** When Spotify's `/audio-features` endpoint returned a 403 error, Claude suggested using OpenAI to estimate the missing features rather than abandoning the Spotify integration entirely. This turned a blocker into an architectural feature — the hybrid approach ended up working well.

**Flawed suggestion:** Claude initially suggested using the Spotify `/recommendations` endpoint as the primary source for populating the song library. This endpoint has been restricted for new apps since November 2024, which Claude's training data did not reflect. The suggestion had to be replaced with `sp.search()` + pagination after testing revealed the 403 error.

### What This Taught Me

Building this project revealed that the hardest part of a recommender system is not the scoring algorithm — it is deciding *what counts as signal*. A completed play, a skip at 20%, a hearted song, and a song that appears in your all-time top 50 all mean very different things. Encoding that meaning into structured events that an LLM can reason about is fundamentally a design problem, not a machine learning one. The model is only as good as the schema you give it to work with.

---

## Stretch Features

### Test Harness (+2)

`src/evaluate_llm_reliability.py` implements a full predefined test harness. Instead of just running on live data, it defines 5 synthetic scenarios with known inputs and expected outputs, then reports pass/fail and confidence ratings for each.

```text
python -m src.evaluate_llm_reliability

  Scenario 1: High-energy listener nudges energy + tempo upward
  Result: PASS  |  Confidence: 100%  |  Elapsed: 0ms
  ...
  SUMMARY: 6 / 6 scenarios passed
  RESULT: PASS
```

**Why this is valuable:** Unit tests check individual functions; the harness checks end-to-end behavioral contracts. Scenario 3 (protected field injection) simulates a prompt-injection attack and verifies the guardrail layer blocks it without needing a real LLM call.

---

### Fine-Tuning / Specialization (+2)

`llm_reeval.py` adds 4 few-shot examples directly to `SYSTEM_PROMPT`. Each example shows a concrete scenario with candidate changes, evidence, and the correct vs incorrect output:

- **Example A** — Accept an energy nudge when completes clearly dominate
- **Example B** — Veto a genre change when evidence is split (minority of completes)
- **Example C** — Never emit `favorite_artist` or other protected fields in the response
- **Example D** — Reduce a proposed delta when skips and completes conflict

This is specialization via prompt engineering (in-context few-shot fine-tuning). The result is a more reliable LLM pass that correctly vetoes weak genre changes and avoids emitting protected fields, even when the deterministic layer proposes them.

**Measured impact:** Before adding examples, the LLM occasionally returned `favorite_genre` even when skips and completes were tied (3 r&b vs 7 hip-hop). After adding Example B, that case correctly returns `{}`.

---

### Agentic Workflow Enhancement (+2)

The startup pipeline now emits observable numbered step logs with timing at every stage. Instead of a flat block of `[sync]` lines, the output shows a 4-stage pipeline (A→D) wrapping the 8-step Spotify sync:

```text
================================================================
  PIPELINE — Startup Sync
================================================================

[A] Retrieving listening data from Spotify (3 sources)...
  [step 1/8] Spotify OAuth authentication ... done (312ms)
  [step 2/8] Fetch recently played tracks ... done (489ms)
         → 50 tracks retrieved
  [step 3/8] Fetch top tracks (short / medium / long term) ... done (1204ms)
         → 150 tracks (50/50/50 short/med/long)
  [step 4/8] Fetch liked songs ... done (421ms)
         → 50 tracks retrieved
  [step 5/8] Detect new tracks + estimate audio features via OpenAI ... done (8231ms)
         → 12 new tracks found
         → songs.csv updated: 228 total (+12)
  [step 6/8] Skip inference from recently-played timestamps ... done (1ms)
         → 8 events: 7 complete, 1 skip
  [step 7/8] Build top-track + liked-song history events ... done (0ms)
         → 0 top-track events, 0 liked events
  [step 8/8] Append new events to history.jsonl ... done (2ms)
         → 8 events written
    → 228 songs in library | 11.2s total

[B] Deterministic analysis of listening history...
    → 75 events analysed | 2 candidate changes | 1ms
       candidate: target_energy = 0.45
       candidate: target_tempo_bpm = 100

[C] LLM refinement (OpenAI gpt-4.1-mini)...
    → method: hybrid (deterministic + OpenAI) | 1.8s

[D] Profile versioning...
    → v14 → v15 saved
    → reason: hybrid — target_energy: 0.6→0.45 | ...
```

This makes every intermediate decision visible: what Spotify returned, what the deterministic layer proposed, what the LLM decided, and what changed in the profile.

---

### RAG Enhancement (+2)

The Spotify sync layer is a form of retrieval-augmented profiling: instead of one history source, it retrieves from **three temporally distinct stores** before running the LLM profile update.

| Source | Temporal window | Signal type | Event weight |
| --- | --- | --- | --- |
| Recently played | Last few hours | Skip / complete inferred from timestamps | Medium |
| Top tracks (short-term) | Last 4 weeks | Completed plays | Medium |
| Top tracks (medium/long-term) | 6 months / all-time | Completed plays | Medium |
| Liked songs | All-time explicit | User deliberately hearted | **Highest** (written as "repeat") |

**Measured impact vs single-source:**

After the first sync against only recently played (8 events), the profile was stuck at the initial `pop/happy` default — not enough signal. After enabling all three sources (50 recently played + 150 top tracks + 50 liked songs = 208 events), the profile updated 14 times in a single day and correctly converged on `hip-hop/sad` with `target_energy=0.6`, matching actual listening behavior.

The liked songs source alone contributed 50 high-confidence `repeat` events — the strongest positive signal — which is what drove the profile away from the default. Without multi-source retrieval, this signal would not exist until the user explicitly replayed songs in the terminal player.

---

## Project Structure

```text
├── src/
│   ├── main.py              # CLI entry point
│   ├── config.py            # Centralized data paths
│   ├── models.py            # Data models (Song, UserProfile, InteractionEvent)
│   ├── recommender.py       # Weighted scoring engine + diversity re-ranker
│   ├── player.py            # Terminal player UI
│   ├── llm_reeval.py        # LLM-driven profile updater
│   ├── spotify_sync.py      # Spotify sync (recently played, top, liked)
│   ├── spotify_fetch.py     # Standalone genre-based library populator
│   ├── spotify_utils.py     # Shared Spotify + OpenAI utilities
│   └── evaluate_llm_reliability.py  # Reliability test harness
├── data/
│   ├── songs.csv            # Song library (216 tracks)
│   ├── history.jsonl        # Interaction event log
│   └── user_profile.json    # Versioned taste profile
├── docs/
│   ├── Pipeline.md          # System pipeline documentation
│   ├── model_card.md        # Model card
│   └── reflection.md        # Module 1–3 reflection
├── tests/
│   ├── test_recommender.py  # Scoring and ranking tests
│   └── test_llm_reeval.py   # LLM re-evaluation tests
└── assets/                  # Diagrams and screenshots
```

---

## Portfolio Reflection

This project reflects how I think about building AI systems: skeptically and incrementally. I didn't trust the LLM to update my music profile freely — I built a deterministic layer first, then let the LLM refine bounded candidates, then added a guardrail that blocks invalid outputs before they reach the profile, then added an impact gate that discards updates that don't actually change what gets recommended. Each layer exists because I found a specific failure mode during testing. That instinct — to constrain AI behavior at every boundary rather than assume it will behave — is what I want to carry into production AI work. The other lesson was about what counts as signal: a completed play, a 20%-skip, a hearted song, and an all-time top track all mean different things, and encoding that meaning into structured events that an LLM can reason about turned out to be harder than any algorithm I wrote. That is the real design problem in recommender systems.
