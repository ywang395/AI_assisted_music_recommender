from src.recommender import (
    Recommender,
    Song,
    UserProfile,
    genre_similarity,
    mood_similarity,
    recommend_songs,
    score_song,
)


def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_for_genre_similarity():
    assert genre_similarity("pop", "indie pop") == 0.8
    assert genre_similarity("rock", "metal") == 0.7
    assert genre_similarity("pop", "classical") == 0.0

    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    }
    similar_song = {
        "genre": "indie pop",
        "mood": "happy",
        "artist": "Another Artist",
        "energy": 0.8,
        "danceability": 0.8,
        "valence": 0.9,
        "acousticness": 0.2,
    }
    unrelated_song = {
        "genre": "classical",
        "mood": "happy",
        "artist": "Another Artist",
        "energy": 0.8,
        "danceability": 0.8,
        "valence": 0.9,
        "acousticness": 0.2,
    }

    similar_score, similar_reasons = score_song(user_prefs, similar_song)
    unrelated_score, _ = score_song(user_prefs, unrelated_song)

    assert similar_score > unrelated_score
    assert any("similar genre match" in reason for reason in similar_reasons)


def test_for_mood_similarity():
    assert mood_similarity("happy", "joyful") == 0.8
    assert mood_similarity("sad", "melancholic") == 0.8
    assert mood_similarity("happy", "angry") == 0.0

    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    }
    similar_song = {
        "genre": "pop",
        "mood": "joyful",
        "artist": "Another Artist",
        "energy": 0.8,
        "danceability": 0.8,
        "valence": 0.9,
        "acousticness": 0.2,
    }
    unrelated_song = {
        "genre": "pop",
        "mood": "angry",
        "artist": "Another Artist",
        "energy": 0.8,
        "danceability": 0.8,
        "valence": 0.9,
        "acousticness": 0.2,
    }

    similar_score, similar_reasons = score_song(user_prefs, similar_song)
    unrelated_score, _ = score_song(user_prefs, unrelated_song)

    assert similar_score > unrelated_score
    assert any("similar mood match" in reason for reason in similar_reasons)


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_score_song_uses_tempo_preference():
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.5,
        "tempo_bpm": 140,
        "likes_acoustic": False,
    }
    fast_song = {
        "genre": "pop",
        "mood": "happy",
        "artist": "Fast Artist",
        "energy": 0.5,
        "tempo_bpm": 138,
        "danceability": 0.7,
        "valence": 0.8,
        "acousticness": 0.2,
    }
    slow_song = {
        "genre": "pop",
        "mood": "happy",
        "artist": "Slow Artist",
        "energy": 0.5,
        "tempo_bpm": 75,
        "danceability": 0.7,
        "valence": 0.8,
        "acousticness": 0.2,
    }

    fast_score, fast_reasons = score_song(user_prefs, fast_song)
    slow_score, _ = score_song(user_prefs, slow_song)

    assert fast_score > slow_score
    assert any("tempo fit" in reason for reason in fast_reasons)


def test_recommender_accepts_csv_path():
    rec = Recommender("data/songs.csv")

    assert rec.songs
    assert isinstance(rec.songs[0], Song)


def test_recommend_songs_penalizes_recently_skipped_specific_song():
    songs = [
        {
            "id": 1,
            "title": "Perfect Match But Skipped",
            "artist": "Test Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.9,
            "danceability": 0.8,
            "acousticness": 0.2,
        },
        {
            "id": 2,
            "title": "Slightly Weaker Fresh Song",
            "artist": "Other Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.72,
            "tempo_bpm": 118,
            "valence": 0.82,
            "danceability": 0.72,
            "acousticness": 0.25,
        },
    ]
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "tempo_bpm": 120,
        "danceability": 0.8,
        "valence": 0.9,
        "likes_acoustic": False,
        "song_penalties": {"1": 0.75},
    }

    skipped_score, _ = score_song(user_prefs, songs[0])
    ranked = recommend_songs(user_prefs, songs, k=2)

    assert skipped_score > 0
    assert ranked[0][0]["id"] == 2
    assert "recent skip penalty" in ranked[1][2]


def test_recommend_songs_prefers_exact_genre_pool_when_enough_matches():
    songs = [
        {
            "id": 1,
            "title": "Hip Hop One",
            "artist": "A",
            "genre": "hip-hop",
            "mood": "intense",
            "energy": 0.8,
            "tempo_bpm": 95,
            "valence": 0.4,
            "danceability": 0.7,
            "acousticness": 0.1,
        },
        {
            "id": 2,
            "title": "Hip Hop Two",
            "artist": "B",
            "genre": "hip-hop",
            "mood": "sad",
            "energy": 0.7,
            "tempo_bpm": 100,
            "valence": 0.4,
            "danceability": 0.6,
            "acousticness": 0.2,
        },
        {
            "id": 3,
            "title": "Hip Hop Three",
            "artist": "C",
            "genre": "hip-hop",
            "mood": "motivated",
            "energy": 0.9,
            "tempo_bpm": 110,
            "valence": 0.5,
            "danceability": 0.8,
            "acousticness": 0.05,
        },
        {
            "id": 4,
            "title": "Perfect Mood Lofi",
            "artist": "D",
            "genre": "lofi",
            "mood": "sad",
            "energy": 0.75,
            "tempo_bpm": 100,
            "valence": 0.4,
            "danceability": 0.7,
            "acousticness": 0.2,
        },
    ]
    user_prefs = {
        "genre": "hip-hop",
        "mood": "sad",
        "energy": 0.75,
        "tempo_bpm": 100,
        "danceability": 0.7,
        "valence": 0.4,
        "likes_acoustic": False,
    }

    ranked = recommend_songs(user_prefs, songs, k=3)

    assert [item[0]["genre"] for item in ranked] == ["hip-hop", "hip-hop", "hip-hop"]


def test_recommend_songs_boosts_artist_without_forcing_artist_only_queue():
    songs = [
        {
            "id": 1,
            "title": "Eminem One",
            "artist": "Eminem",
            "genre": "hip-hop",
            "mood": "intense",
            "energy": 0.85,
            "tempo_bpm": 90,
            "valence": 0.4,
            "danceability": 0.7,
            "acousticness": 0.05,
        },
        {
            "id": 2,
            "title": "Eminem Two",
            "artist": "Eminem",
            "genre": "hip-hop",
            "mood": "motivated",
            "energy": 0.9,
            "tempo_bpm": 88,
            "valence": 0.5,
            "danceability": 0.75,
            "acousticness": 0.02,
        },
        {
            "id": 3,
            "title": "Perfect Non Artist",
            "artist": "Other Artist",
            "genre": "hip-hop",
            "mood": "sad",
            "energy": 0.75,
            "tempo_bpm": 100,
            "valence": 0.4,
            "danceability": 0.7,
            "acousticness": 0.2,
        },
    ]
    user_prefs = {
        "genre": "hip-hop",
        "artist": "Eminem",
        "mood": "sad",
        "energy": 0.75,
        "tempo_bpm": 100,
        "danceability": 0.7,
        "valence": 0.4,
        "likes_acoustic": False,
    }

    ranked = recommend_songs(user_prefs, songs, k=2)

    assert ranked[0][0]["artist"] == "Other Artist"
    assert ranked[1][0]["artist"] == "Eminem"
    assert "artist match" in ranked[1][2]
