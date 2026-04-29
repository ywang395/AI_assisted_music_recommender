from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union


def _normalize_text(value: str) -> str:
    return str(value).strip().lower()


def _clamp_similarity(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_tags(value: str) -> List[str]:
    if not value:
        return []
    return [_normalize_text(tag) for tag in str(value).split("|") if _normalize_text(tag)]


GENRE_SIMILARITY = {
    ("pop", "indie pop"): 0.8,
    ("indie pop", "pop"): 0.8,
    ("rock", "metal"): 0.7,
    ("metal", "rock"): 0.7,
    ("lofi", "ambient"): 0.7,
    ("ambient", "lofi"): 0.7,
    ("jazz", "blues"): 0.6,
    ("blues", "jazz"): 0.6,
    ("r&b", "hip-hop"): 0.6,
    ("hip-hop", "r&b"): 0.6,
}


def genre_similarity(user_genre: str, song_genre: str) -> float:
    user_genre = _normalize_text(user_genre)
    song_genre = _normalize_text(song_genre)

    if user_genre == song_genre:
        return 1.0

    return GENRE_SIMILARITY.get((user_genre, song_genre), 0.0)


MOOD_SIMILARITY = {
    ("happy", "joyful"): 0.8,
    ("joyful", "happy"): 0.8,
    ("sad", "melancholic"): 0.8,
    ("melancholic", "sad"): 0.8,
    ("chill", "relaxed"): 0.7,
    ("relaxed", "chill"): 0.7,
    ("chill", "peaceful"): 0.7,
    ("peaceful", "chill"): 0.7,
    ("focused", "motivated"): 0.6,
    ("motivated", "focused"): 0.6,
    ("dreamy", "romantic"): 0.6,
    ("romantic", "dreamy"): 0.6,
    ("intense", "energetic"): 0.7,
    ("energetic", "intense"): 0.7,
}


def mood_similarity(user_mood: str, song_mood: str) -> float:
    user_mood = _normalize_text(user_mood)
    song_mood = _normalize_text(song_mood)

    if user_mood == song_mood:
        return 1.0

    return MOOD_SIMILARITY.get((user_mood, song_mood), 0.0)


SCORING_MODES = {
    "balanced": {
        "genre": 0.16,
        "mood": 0.14,
        "energy": 0.12,
        "tempo": 0.06,
        "danceability": 0.10,
        "valence": 0.06,
        "acousticness": 0.07,
        "popularity": 0.08,
        "decade": 0.05,
        "mood_tags": 0.08,
        "live_energy": 0.04,
        "lyrical_depth": 0.04,
        "instrumentalness": 0.03,
    },
    "genre_first": {
        "genre": 0.26,
        "mood": 0.10,
        "energy": 0.08,
        "tempo": 0.05,
        "danceability": 0.07,
        "valence": 0.05,
        "acousticness": 0.06,
        "popularity": 0.06,
        "decade": 0.09,
        "mood_tags": 0.08,
        "live_energy": 0.04,
        "lyrical_depth": 0.05,
        "instrumentalness": 0.05,
    },
    "mood_first": {
        "genre": 0.10,
        "mood": 0.24,
        "energy": 0.08,
        "tempo": 0.05,
        "danceability": 0.08,
        "valence": 0.08,
        "acousticness": 0.06,
        "popularity": 0.05,
        "decade": 0.05,
        "mood_tags": 0.16,
        "live_energy": 0.04,
        "lyrical_depth": 0.04,
        "instrumentalness": 0.02,
    },
    "energy_focused": {
        "genre": 0.09,
        "mood": 0.09,
        "energy": 0.26,
        "tempo": 0.10,
        "danceability": 0.12,
        "valence": 0.07,
        "acousticness": 0.05,
        "popularity": 0.05,
        "decade": 0.04,
        "mood_tags": 0.06,
        "live_energy": 0.08,
        "lyrical_depth": 0.02,
        "instrumentalness": 0.03,
    },
}


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: int = 50
    release_decade: int = 2010
    mood_tags: str = ""
    live_energy: float = 0.5
    lyrical_depth: float = 0.5
    instrumentalness: float = 0.5


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    favorite_artist: str = ""
    target_danceability: float = 0.5
    target_valence: float = 0.5
    target_tempo_bpm: int = 100
    desired_popularity: float = 0.5
    preferred_decade: int = 2010
    preferred_mood_tags: List[str] = field(default_factory=list)
    target_live_energy: float = 0.5
    target_lyrical_depth: float = 0.5
    target_instrumentalness: float = 0.5
    scoring_mode: str = "balanced"


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: Union[str, List[Song], List[Dict]]):
        # Accept a CSV path for convenience in scripts, while still supporting
        # direct Song objects in tests and app code.
        if isinstance(songs, str):
            raw_songs = load_songs(songs)
        else:
            raw_songs = songs

        self.songs = [self._coerce_song(song) for song in raw_songs]

    @staticmethod
    def _coerce_song(song: Union[Song, Dict]) -> Song:
        if isinstance(song, Song):
            return song

        return Song(
            id=int(song["id"]),
            title=song["title"],
            artist=song["artist"],
            genre=song["genre"],
            mood=song["mood"],
            energy=float(song["energy"]),
            tempo_bpm=float(song["tempo_bpm"]),
            valence=float(song["valence"]),
            danceability=float(song["danceability"]),
            acousticness=float(song["acousticness"]),
            popularity=int(song.get("popularity", 50)),
            release_decade=int(song.get("release_decade", 2010)),
            mood_tags=song.get("mood_tags", ""),
            live_energy=float(song.get("live_energy", 0.5)),
            lyrical_depth=float(song.get("lyrical_depth", 0.5)),
            instrumentalness=float(song.get("instrumentalness", 0.5)),
        )

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = {
            "genre": user.favorite_genre,
            "artist": user.favorite_artist,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "danceability": user.target_danceability,
            "valence": user.target_valence,
            "tempo_bpm": user.target_tempo_bpm,
            "popularity": user.desired_popularity,
            "release_decade": user.preferred_decade,
            "mood_tags": user.preferred_mood_tags,
            "live_energy": user.target_live_energy,
            "lyrical_depth": user.target_lyrical_depth,
            "instrumentalness": user.target_instrumentalness,
        }
        ranked = recommend_songs(user_prefs, [song.__dict__ for song in self.songs], k=k, mode=user.scoring_mode)
        by_id = {song.id: song for song in self.songs}
        return [by_id[item["id"]] for item, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = {
            "genre": user.favorite_genre,
            "artist": user.favorite_artist,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "danceability": user.target_danceability,
            "valence": user.target_valence,
            "tempo_bpm": user.target_tempo_bpm,
            "popularity": user.desired_popularity,
            "release_decade": user.preferred_decade,
            "mood_tags": user.preferred_mood_tags,
            "live_energy": user.target_live_energy,
            "lyrical_depth": user.target_lyrical_depth,
            "instrumentalness": user.target_instrumentalness,
        }
        score, reasons = score_song(user_prefs, song.__dict__, mode=user.scoring_mode)
        return f"Score {score:.2f}: " + ", ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    import csv

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": _normalize_text(row["genre"]),
                    "mood": _normalize_text(row["mood"]),
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                    "popularity": int(row.get("popularity", 50)),
                    "release_decade": int(row.get("release_decade", 2010)),
                    "mood_tags": row.get("mood_tags", ""),
                    "live_energy": float(row.get("live_energy", 0.5)),
                    "lyrical_depth": float(row.get("lyrical_depth", 0.5)),
                    "instrumentalness": float(row.get("instrumentalness", 0.5)),
                }
            )
        print(f"Loaded songs: {len(songs)}")
    return songs


def _numeric_similarity(song_value: float, user_value: float, scale: float = 1.0) -> float:
    return _clamp_similarity(1.0 - abs(song_value - user_value) / scale)


def _mood_tag_similarity(user_tags: List[str], song_tags: List[str]) -> float:
    if not user_tags:
        return 0.5
    overlap = len(set(user_tags) & set(song_tags))
    return overlap / len(set(user_tags))


def _get_weights(mode: str, user_artist: str) -> Dict[str, float]:
    weights = dict(SCORING_MODES.get(mode, SCORING_MODES["balanced"]))
    if user_artist:
        weights["artist"] = 0.12
        weights["popularity"] = max(0.0, weights["popularity"] - 0.04)
        weights["instrumentalness"] = max(0.0, weights["instrumentalness"] - 0.04)
    return weights


def _build_reasons(component_scores: Dict[str, float], weights: Dict[str, float], user_artist: str) -> List[str]:
    def fmt(label: str, points: float) -> str:
        return f"{label} (+{points:.2f})"

    reasons = []
    if component_scores["genre"] == 1.0:
        reasons.append(fmt("genre match", weights["genre"] * component_scores["genre"]))
    elif component_scores["genre"] > 0.0:
        reasons.append(fmt("similar genre match", weights["genre"] * component_scores["genre"]))
    if component_scores["mood"] == 1.0:
        reasons.append(fmt("mood match", weights["mood"] * component_scores["mood"]))
    elif component_scores["mood"] > 0.0:
        reasons.append(fmt("similar mood match", weights["mood"] * component_scores["mood"]))
    if component_scores["mood_tags"] > 0.0:
        reasons.append(fmt("mood tag fit", weights["mood_tags"] * component_scores["mood_tags"]))
    if user_artist and component_scores["artist"]:
        reasons.append(fmt("artist match", weights["artist"] * component_scores["artist"]))
    reasons.append(fmt("energy similarity", weights["energy"] * component_scores["energy"]))
    reasons.append(fmt("tempo fit", weights["tempo"] * component_scores["tempo"]))
    reasons.append(fmt("danceability similarity", weights["danceability"] * component_scores["danceability"]))
    reasons.append(fmt("valence similarity", weights["valence"] * component_scores["valence"]))
    reasons.append(fmt("popularity fit", weights["popularity"] * component_scores["popularity"]))
    reasons.append(fmt("era fit", weights["decade"] * component_scores["decade"]))
    reasons.append(fmt("live energy fit", weights["live_energy"] * component_scores["live_energy"]))
    reasons.append(fmt("lyrical depth fit", weights["lyrical_depth"] * component_scores["lyrical_depth"]))
    reasons.append(fmt("instrumentalness fit", weights["instrumentalness"] * component_scores["instrumentalness"]))
    return reasons


def score_song(user_prefs: Dict, song: Dict, mode: str = "balanced") -> Tuple[float, List[str]]:
    user_genre = _normalize_text(user_prefs.get("genre", ""))
    user_artist = _normalize_text(user_prefs.get("artist", ""))
    user_mood = _normalize_text(user_prefs.get("mood", ""))
    user_energy = float(user_prefs.get("energy", 0.5))
    user_tempo = float(user_prefs.get("tempo_bpm", 100.0))
    user_danceability = float(user_prefs.get("danceability", 0.5))
    user_valence = float(user_prefs.get("valence", 0.5))
    user_popularity = float(user_prefs.get("popularity", 0.5))
    user_decade = int(user_prefs.get("release_decade", 2010))
    raw_tags = user_prefs.get("mood_tags", [])
    user_tags = _parse_tags("|".join(raw_tags)) if isinstance(raw_tags, list) else _parse_tags(raw_tags)
    user_live_energy = float(user_prefs.get("live_energy", 0.5))
    user_lyrical_depth = float(user_prefs.get("lyrical_depth", 0.5))
    user_instrumentalness = float(user_prefs.get("instrumentalness", 0.5))
    likes_acoustic = user_prefs.get("likes_acoustic", None)

    song_genre = _normalize_text(song.get("genre", ""))
    song_artist = _normalize_text(song.get("artist", ""))
    song_mood = _normalize_text(song.get("mood", ""))
    song_energy = float(song.get("energy", 0.5))
    song_tempo = float(song.get("tempo_bpm", 100.0))
    song_danceability = float(song.get("danceability", 0.5))
    song_valence = float(song.get("valence", 0.5))
    song_acousticness = float(song.get("acousticness", 0.5))
    song_popularity = float(song.get("popularity", 50)) / 100.0  # normalize 0-100 → 0-1 to match user_popularity
    song_decade = int(song.get("release_decade", 2010))
    song_tags = _parse_tags(song.get("mood_tags", ""))
    song_live_energy = float(song.get("live_energy", 0.5))
    song_lyrical_depth = float(song.get("lyrical_depth", 0.5))
    song_instrumentalness = float(song.get("instrumentalness", 0.5))

    component_scores = {
        "genre": genre_similarity(user_genre, song_genre),
        "mood": mood_similarity(user_mood, song_mood),
        "artist": 1.0 if user_artist and song_artist == user_artist else 0.0,
        "energy": _numeric_similarity(song_energy, user_energy),
        "tempo": _numeric_similarity(song_tempo, user_tempo, scale=100.0),
        "danceability": _numeric_similarity(song_danceability, user_danceability),
        "valence": _numeric_similarity(song_valence, user_valence),
        "popularity": _numeric_similarity(song_popularity, user_popularity),
        "decade": _numeric_similarity(song_decade, user_decade, scale=40.0),
        "mood_tags": _mood_tag_similarity(user_tags, song_tags),
        "live_energy": _numeric_similarity(song_live_energy, user_live_energy),
        "lyrical_depth": _numeric_similarity(song_lyrical_depth, user_lyrical_depth),
        "instrumentalness": _numeric_similarity(song_instrumentalness, user_instrumentalness),
    }

    weights = _get_weights(mode, user_artist)

    if likes_acoustic is None:
        component_scores["acousticness"] = 0.5
        acousticness_reason = f"acousticness neutral (+{weights['acousticness'] * 0.5:.2f})"
    else:
        component_scores["acousticness"] = song_acousticness if likes_acoustic else (1.0 - song_acousticness)
        acoustic_label = "acousticness fit" if likes_acoustic else "low acousticness fit"
        acousticness_reason = f"{acoustic_label} (+{weights['acousticness'] * component_scores['acousticness']:.2f})"

    score = sum(weights[key] * component_scores[key] for key in weights if key in component_scores)
    reasons = _build_reasons(component_scores, weights, user_artist)
    reasons.append(acousticness_reason)
    return score, reasons


def _apply_diversity_penalty(song: Dict, selected_songs: List[Dict]) -> Tuple[float, List[str]]:
    penalty = 0.0
    reasons = []
    if any(_normalize_text(existing.get("artist", "")) == _normalize_text(song.get("artist", "")) for existing in selected_songs):
        penalty += 0.08
        reasons.append("artist diversity penalty (-0.08)")
    if any(_normalize_text(existing.get("genre", "")) == _normalize_text(song.get("genre", "")) for existing in selected_songs):
        penalty += 0.05
        reasons.append("genre diversity penalty (-0.05)")
    return penalty, reasons


def _history_penalty(song: Dict, user_prefs: Dict) -> Tuple[float, List[str]]:
    penalties = user_prefs.get("song_penalties", {}) or {}
    song_id = song.get("id", "")
    raw_penalty = penalties.get(song_id, penalties.get(str(song_id), 0.0))
    penalty = max(0.0, min(0.75, float(raw_penalty or 0.0)))
    if penalty <= 0:
        return 0.0, []
    return penalty, [f"recent skip penalty (-{penalty:.2f})"]


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = "balanced"
) -> List[Tuple[Dict, float, str]]:
    candidate_songs = songs
    preferred_genre = _normalize_text(user_prefs.get("genre", ""))
    exact_genre_matches = [
        song for song in candidate_songs
        if preferred_genre and _normalize_text(song.get("genre", "")) == preferred_genre
    ]
    candidate_songs = exact_genre_matches if len(exact_genre_matches) >= k else candidate_songs

    scored = []
    for song in candidate_songs:
        score, reasons = score_song(user_prefs, song, mode=mode)
        penalty, penalty_reasons = _history_penalty(song, user_prefs)
        score = max(0.0, score - penalty)
        reasons = reasons + penalty_reasons
        scored.append({"song": song, "base_score": score, "reasons": reasons})

    scored.sort(key=lambda item: item["base_score"], reverse=True)

    selected = []
    selected_songs = []
    while scored and len(selected) < k:
        best_item = None
        best_final_score = float("-inf")
        best_penalty_reasons: List[str] = []
        for item in scored:
            penalty, penalty_reasons = _apply_diversity_penalty(item["song"], selected_songs)
            final_score = item["base_score"] - penalty
            if final_score > best_final_score:
                best_final_score = final_score
                best_item = item
                best_penalty_reasons = penalty_reasons

        if best_item is None:
            break

        scored.remove(best_item)
        selected_songs.append(best_item["song"])
        all_reasons = best_item["reasons"] + best_penalty_reasons
        selected.append((best_item["song"], best_final_score, ", ".join(all_reasons)))

    return selected
