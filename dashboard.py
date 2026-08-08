import streamlit as st
from transformers import pipeline
import shap

from emotion_utils import MODEL_NAME, get_top_emotion

MAX_CHARS = 512

st.set_page_config(page_title="MindScan XAI", page_icon="🧠", layout="wide")
st.title("🧠 MindScan: Explainable AI (XAI) Pattern Analyzer")
st.write(
    "This tool uses **DistilBERT** for emotion detection and **SHAP** to explain word-level influence."
)
st.warning(
    "⚠️ **Disclaimer:** MindScan is a portfolio/research project for demonstrating "
    "explainable AI. It is **not** a diagnostic or clinical tool and must not be used "
    "as a substitute for professional mental health advice. If you or someone you know "
    "is in crisis, please contact a local emergency service or a helpline such as "
    "**iCall (+91 9152987821)** or **AASRA (+91 9820466726)**."
)


@st.cache_resource(show_spinner="Loading DistilBERT emotion model...")
def load_assets():
    """Load the classification pipeline and its SHAP explainer once per session."""
    pipe = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    explainer = shap.Explainer(pipe)
    return pipe, explainer


try:
    classifier, explainer = load_assets()
    model_ready = True
except Exception as exc:
    model_ready = False
    st.error(f"Failed to load the model. Please check your internet connection or dependencies.\n\n`{exc}`")

user_input = st.text_area(
    "Enter text for deep analysis:",
    height=150,
    max_chars=MAX_CHARS,
    placeholder="Try the sarcasm line here...",
)

if st.button("Deep Pattern Analysis", disabled=not model_ready):
    if not user_input or not user_input.strip():
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            col1, col2 = st.columns(2)

            # --- Column 1: The 'What' (Predictions) ---
            results = classifier(user_input)[0]
            top_emotion = get_top_emotion(results)

            with col1:
                st.subheader("Emotional Distribution")
                for res in sorted(results, key=lambda r: r["score"], reverse=True):
                    st.write(f"**{res['label'].upper()}**")
                    st.progress(res["score"])

            # --- Column 2: The 'Why' (Explainability) ---
            with col2:
                st.subheader("Word-Level Influence (XAI)")
                st.write(
                    f"Explaining the top predicted emotion: **{top_emotion['label'].upper()}** "
                    f"({top_emotion['score']:.1%} confidence)"
                )
                st.write("Red = Increases emotion | Blue = Decreases emotion")

                try:
                    shap_values = explainer([user_input])
                    fig = shap.plots.text(
                        shap_values[0, :, top_emotion["label"]], display=False
                    )
                    st.components.v1.html(fig, height=200, scrolling=True)
                    st.info(
                        "The highlighted words above show exactly which terms forced "
                        f"the model toward its **{top_emotion['label'].upper()}** conclusion."
                    )
                except Exception as exc:
                    st.error(f"Could not generate the SHAP explanation for this input.\n\n`{exc}`")
