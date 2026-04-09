import streamlit as st
from transformers import pipeline
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="MindScan XAI", page_icon="🧠", layout="wide")
st.title("🧠 MindScan: Explainable AI (XAI) Pattern Analyzer")
st.write("This tool uses **DistilBERT** for emotion detection and **SHAP** to explain word-level influence.")

# 1. Load Model and SHAP Explainer
@st.cache_resource
def load_assets():
    # We use return_all_scores to get the full distribution
    pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
    # SHAP explainer helps us see 'why'
    explainer = shap.Explainer(pipe)
    return pipe, explainer

classifier, explainer = load_assets()

# 2. User Input
user_input = st.text_area("Enter text for deep analysis:", height=150, placeholder="Try the sarcasm line here...")

if st.button("Deep Pattern Analysis"):
    if user_input:
        col1, col2 = st.columns(2)
        
        # --- Column 1: The 'What' (Predictions) ---
        with col1:
            st.subheader("Emotional Distribution")
            results = classifier(user_input)[0]
            for res in results:
                st.write(f"**{res['label'].upper()}**")
                st.progress(res['score'])

        # --- Column 2: The 'Why' (Explainability) ---
        with col2:
            st.subheader("Word-Level Influence (XAI)")
            st.write("Red = Increases emotion | Blue = Decreases emotion")
            
            # Generate SHAP values
            shap_values = explainer([user_input])
            
            # This creates a 'Force Plot' or 'Text Plot'
            # For simplicity in Streamlit, we'll use SHAP's built-in text renderer
            st.write("Analysis of dominant triggers:")
            fig = shap.plots.text(shap_values[0, :, "joy"], display=False) 
            st.components.v1.html(fig, height=200, scrolling=True)
            
            st.info("The highlighted words above show exactly which terms forced the model toward its primary conclusion.")
    else:
        st.warning("Please enter text to analyze.")