import pytest

from emotion_utils import get_top_emotion


def test_get_top_emotion_returns_highest_score():
    results = [
        {"label": "joy", "score": 0.1},
        {"label": "anger", "score": 0.7},
        {"label": "sadness", "score": 0.2},
    ]
    top = get_top_emotion(results)
    assert top["label"] == "anger"
    assert top["score"] == 0.7


def test_get_top_emotion_single_result():
    results = [{"label": "fear", "score": 0.99}]
    assert get_top_emotion(results)["label"] == "fear"


def test_get_top_emotion_ignores_input_order():
    results = [
        {"label": "surprise", "score": 0.4},
        {"label": "love", "score": 0.4001},
    ]
    assert get_top_emotion(results)["label"] == "love"


def test_get_top_emotion_empty_raises():
    with pytest.raises(ValueError):
        get_top_emotion([])
