"""Lightweight CLI demo of the emotion classifier (no SHAP explanation).

For the full explainable-AI dashboard, run: streamlit run dashboard.py
"""
from transformers import pipeline

from emotion_utils import MODEL_NAME


def main():
    print("Loading the AI model... please wait.")
    classifier = pipeline("text-classification", model=MODEL_NAME)

    user_text = input("\nDescribe how you are feeling today: ").strip()
    if not user_text:
        print("No input provided, exiting.")
        return

    result = classifier(user_text)[0]
    print("\n--- Analysis ---")
    print(f"Detected Emotion: {result['label'].upper()}")
    print(f"Confidence: {result['score']:.2%}")


if __name__ == "__main__":
    main()
