"""Pure helper functions shared by the CLI (app.py) and the dashboard (dashboard.py).

Kept dependency-free (no torch/transformers/streamlit imports) so it stays
fast and easy to unit test.
"""

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"


def get_top_emotion(results):
    """Return the prediction dict with the highest confidence score.

    `results` is the list of {"label": str, "score": float} dicts returned by
    the Hugging Face text-classification pipeline for a single input.
    """
    if not results:
        raise ValueError("results must be a non-empty list of prediction dicts")
    return max(results, key=lambda r: r["score"])
