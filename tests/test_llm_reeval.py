import builtins
import sys
import types as py_types

from src.llm_reeval import (
    LLM_EVENT_LIMIT,
    build_llm_prompt,
    call_llm_reeval,
    deterministic_update,
    select_relevant_history_for_llm,
    summarize_recommendation_shift,
)
from src.models import StableUserProfile


def make_profile() -> StableUserProfile:
    return StableUserProfile(
        favorite_genre="pop",
        favorite_mood="sad",
        target_energy=0.4,
        likes_acoustic=True,
        target_danceability=0.1,
        target_valence=0.1,
        desired_popularity=0.85,
        preferred_decade=2010,
    )


def make_history() -> list[dict]:
    return [
        {
            "event_type": "complete",
            "song_title": "Quiet Harbor",
            "song_artist": "Pale Shore",
            "song_genre": "ambient",
            "song_mood": "peaceful",
            "song_energy": 0.2,
            "elapsed_ratio": 1.0,
            "repeat_count": 0,
        }
    ]


def make_large_history(size: int) -> list[dict]:
    history = []
    for index in range(size):
        event_type = "skip" if index % 3 == 0 else ("repeat" if index % 3 == 1 else "complete")
        history.append(
            {
                "event_type": event_type,
                "song_title": f"Song {index}",
                "song_artist": "Test Artist",
                "song_genre": "ambient",
                "song_mood": "peaceful",
                "song_energy": 0.2,
                "elapsed_ratio": 0.1 if event_type == "skip" else 1.0,
                "repeat_count": 1 if event_type == "repeat" else 0,
            }
        )
    return history


def install_fake_openai(monkeypatch, response_text=None, error=None):
    class FakeCompletions:
        def create(self, *, model, messages, response_format, max_tokens):
            assert model == "gpt-4.1-mini"
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert response_format == {"type": "json_object"}
            assert max_tokens == 512
            if error is not None:
                raise error
            return py_types.SimpleNamespace(
                choices=[
                    py_types.SimpleNamespace(
                        message=py_types.SimpleNamespace(content=response_text)
                    )
                ]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.chat = FakeChat()

    fake_openai_module = py_types.ModuleType("openai")
    fake_openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)


def test_call_llm_reeval_without_api_key_falls_back_to_deterministic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    stable = make_profile()
    candidate, method = call_llm_reeval(
        make_history(),
        stable,
        {"target_energy": 0.25},
        allow_genre_change=False,
        allow_mood_change=False,
    )

    assert method == "deterministic only (no API key)"
    assert candidate.target_energy == 0.25


def test_call_llm_reeval_without_google_genai_package_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "openai", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("openai unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    stable = make_profile()
    candidate, method = call_llm_reeval(
        make_history(),
        stable,
        {"target_energy": 0.25},
        allow_genre_change=False,
        allow_mood_change=False,
    )

    assert method == "deterministic only (openai not installed)"
    assert candidate.target_energy == 0.25


def test_call_llm_reeval_on_api_error_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    install_fake_openai(
        monkeypatch,
        error=RuntimeError("invalid API_KEY credentials"),
    )

    stable = make_profile()
    candidate, method = call_llm_reeval(
        make_history(),
        stable,
        {"target_energy": 0.25},
        allow_genre_change=False,
        allow_mood_change=False,
    )

    assert method == "deterministic only (API error)"
    assert candidate.target_energy == 0.25


def test_call_llm_reeval_uses_llm_json_response_when_available(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    install_fake_openai(
        monkeypatch,
        response_text='{"target_energy": 0.3, "favorite_genre": "ambient"}',
    )

    stable = make_profile()
    candidate, method = call_llm_reeval(
        make_history(),
        stable,
        {"target_energy": 0.25, "favorite_genre": "ambient"},
        allow_genre_change=True,
        allow_mood_change=False,
    )

    assert method == "hybrid (deterministic + OpenAI)"
    assert candidate.target_energy == 0.3
    assert candidate.favorite_genre == "ambient"


def test_build_llm_prompt_limits_selected_events_and_mentions_skip_signal():
    prompt = build_llm_prompt(
        make_large_history(40),
        make_profile(),
        {"target_energy": 0.25},
    )

    assert f"max {LLM_EVENT_LIMIT}" in prompt
    assert "negative evidence" in prompt
    assert "Song 0" not in prompt
    assert "Song 39" in prompt


def test_select_relevant_history_for_llm_prioritizes_early_skips_and_respects_limit():
    history = []
    for index in range(30):
        history.append(
            {
                "event_type": "skip" if index < 10 else ("repeat" if index < 15 else "complete"),
                "song_title": f"Song {index}",
                "song_artist": "Test Artist",
                "song_genre": "ambient",
                "song_mood": "peaceful",
                "song_energy": 0.2,
                "elapsed_ratio": 0.1 if index < 6 else 0.8,
                "repeat_count": 1 if 10 <= index < 15 else 0,
            }
        )

    selected = select_relevant_history_for_llm(history, limit=10)

    assert len(selected) == 10
    assert selected == sorted(selected, key=lambda event: int(event["song_title"].split()[-1]))
    assert any(event["song_title"] == "Song 0" for event in selected)
    assert any(event["song_title"] == "Song 5" for event in selected)


def test_summarize_recommendation_shift_reports_when_top_recommendations_change():
    before = make_profile()
    after = make_profile()
    after.favorite_genre = "ambient"
    after.target_tempo_bpm = 120

    summary, changed = summarize_recommendation_shift(before, after, songs_path="data/songs.csv", k=3)

    assert isinstance(summary, str)
    assert "top3" in summary
    assert isinstance(changed, bool)


def test_deterministic_update_moves_from_heavily_skipped_current_genre_to_alternative():
    profile = make_profile()
    profile.favorite_genre = "pop"

    history = (
        [
            {
                "event_type": "skip",
                "song_genre": "pop",
                "song_mood": "happy",
                "song_energy": 0.5,
                "song_tempo_bpm": 100,
                "elapsed_ratio": 0.1,
            }
            for _ in range(8)
        ]
        + [
            {
                "event_type": "complete",
                "song_genre": "pop",
                "song_mood": "happy",
                "song_energy": 0.5,
                "song_tempo_bpm": 100,
                "elapsed_ratio": 1.0,
            }
            for _ in range(6)
        ]
        + [
            {
                "event_type": "complete",
                "song_genre": "hip-hop",
                "song_mood": "sad",
                "song_energy": 0.65,
                "song_tempo_bpm": 95,
                "elapsed_ratio": 1.0,
            }
            for _ in range(4)
        ]
    )

    changes, allow_genre, _ = deterministic_update(history, profile)

    assert allow_genre is True
    assert changes["favorite_genre"] == "hip-hop"


def test_deterministic_update_learns_repeated_positive_artist():
    profile = make_profile()
    profile.favorite_artist = ""
    history = [
        {
            "event_type": "complete",
            "song_artist": "Eminem",
            "song_genre": "hip-hop",
            "song_mood": "intense",
            "song_energy": 0.85,
            "song_tempo_bpm": 90,
            "elapsed_ratio": 1.0,
        },
        {
            "event_type": "repeat",
            "song_artist": "Eminem",
            "song_genre": "hip-hop",
            "song_mood": "intense",
            "song_energy": 0.9,
            "song_tempo_bpm": 88,
            "elapsed_ratio": 1.0,
        },
        {
            "event_type": "complete",
            "song_artist": "Eminem",
            "song_genre": "hip-hop",
            "song_mood": "motivated",
            "song_energy": 0.85,
            "song_tempo_bpm": 90,
            "elapsed_ratio": 1.0,
        },
        {
            "event_type": "complete",
            "song_artist": "Logic",
            "song_genre": "hip-hop",
            "song_mood": "intense",
            "song_energy": 0.85,
            "song_tempo_bpm": 160,
            "elapsed_ratio": 1.0,
        },
    ]

    changes, _, _ = deterministic_update(history, profile)

    assert changes["favorite_artist"] == "Eminem"
