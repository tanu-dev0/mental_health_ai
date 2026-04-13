# MindScan XAI: Explainable Mental Health Pattern Analyzer 

MindScan is a real-time Natural Language Processing (NLP) dashboard designed to identify emotional markers in unstructured text. By utilizing Transformer-based models and Explainable AI (XAI) techniques, it provides transparency into how AI categorizes human emotion.

## Key Features
- **Emotion Classification:** Utilizes a fine-tuned `DistilBERT` model to detect 6 core emotions: Joy, Sadness, Anger, Fear, Love, and Surprise.
- **Explainable AI (XAI):** Integrated with **SHAP (Shapley Additive Explanations)** to visualize word-level feature importance.
- **Interactive Dashboard:** Built with **Streamlit** for real-time inference and visualization.
- **Interpretability:** Highlighting mechanism to show which specific tokens (words) drive the model's emotional verdict.

## Tech Stack
- **Language:** Python 3.14+
- **Deep Learning:** Hugging Face Transformers, PyTorch
- **Interpretability:** SHAP
- **Frontend:** Streamlit
- **Data Visualization:** Matplotlib

## How it Works
The system follows a three-stage pipeline:
1. **Preprocessing:** Tokenization of user input for the DistilBERT transformer.
2. **Inference:** Calculation of probability distributions across emotional classes.
3. **Explanation:** Computation of SHAP values to reveal the contribution of each word toward the final classification.

## Sarcasm Detection Case Study
During development, the model was tested for sarcasm robustness. Through SHAP visualizations, we identified a **lexical bias** where positive adjectives (e.g., "wonderful") override negative contextual intent (e.g., "Wi-Fi crashed"). This finding provides a baseline for future research into contextual linguistic nuance.

## Installation & Usage
1. Clone the repo: `git clone https://github.com/tanu-dev0/mental_health_ai.git`
2. Create venv: `python -m venv venv`
3. Activate: `.\venv\Scripts\activate`
4. Install: `python -m pip install -r requirements.txt`
5. Run: `streamlit run dashboard.py`