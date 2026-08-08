# MindScan XAI: Explainable Mental Health Pattern Analyzer

MindScan is a real-time Natural Language Processing (NLP) dashboard that identifies emotional
markers in unstructured text. It pairs a Transformer-based emotion classifier with Explainable
AI (XAI) so that every prediction comes with a visible, word-level reason — not just a label.

> **Disclaimer:** This is a portfolio/research project demonstrating explainable AI techniques.
> It is **not** a diagnostic or clinical tool and should never replace professional mental health
> support. If you or someone you know is in crisis, contact a local emergency service or a
> helpline such as **iCall (+91 9152987821)** or **AASRA (+91 9820466726)**.

## Key Features
- **Emotion Classification:** Fine-tuned `DistilBERT` model detecting 6 core emotions — joy,
  sadness, anger, fear, love, and surprise.
- **Explainable AI (XAI):** SHAP (Shapley Additive Explanations) visualizes which words pushed
  the model toward its *actual* top-predicted emotion (not a fixed class).
- **Interactive Dashboard:** Built with Streamlit for real-time inference and visualization.
- **Graceful failure handling:** Model-load and explanation errors are caught and surfaced in
  the UI instead of crashing the app.

## Tech Stack
| Layer              | Choice                                   |
|---------------------|-------------------------------------------|
| Language            | Python 3.14                                |
| Deep Learning        | Hugging Face Transformers, PyTorch        |
| Interpretability     | SHAP (Partition Explainer)                |
| Frontend             | Streamlit                                 |
| Testing              | pytest                                     |

## Architecture
```
                 ┌────────────────────┐
 user text  ───▶ │  DistilBERT         │──▶ emotion probabilities (6 classes)
                 │  (HF pipeline)       │
                 └─────────┬───────────┘
                            │ top predicted label
                            ▼
                 ┌────────────────────┐
                 │  SHAP Partition      │──▶ per-token contribution to that label
                 │  Explainer            │
                 └────────────────────┘
```
`emotion_utils.py` holds framework-free helper logic (e.g. picking the top emotion) shared by
both entry points:
- `dashboard.py` — the full Streamlit app with SHAP explanations.
- `app.py` — a minimal CLI demo (classification only, no SHAP) for quick sanity checks.

## Known Limitations & Future Work
- **Lexical bias on sarcasm:** During testing, the model over-weighted positive adjectives
  (e.g. "wonderful") even when the surrounding context was negative (e.g. "the Wi-Fi crashed").
  SHAP made this failure mode visible instead of hiding it — a case study in why explainability
  matters for trust, not just accuracy.
- **Lexical bias on caring/concern vs. romance:** The input *"i am not really in a mood to know
  what i am feeling. i want to comfort my friend but i don't know how because he has fever"* — a
  sentence about platonic worry for a sick friend — was classified as **LOVE (95.3% confidence)**.
  SHAP's word-level view showed "comfort" and "friend" dominating the score, confirming the model
  associates those tokens with romantic love regardless of the surrounding caring-not-romantic
  context. Another concrete example of why raw confidence scores shouldn't be trusted without an
  explanation layer.
- **Single-sentence context window:** Long-form or multi-turn text isn't modeled; each request
  is scored independently.
- **6-class label set:** The base model doesn't cover neutral/mixed emotional states.
- Planned: confidence-threshold flag for low-certainty predictions, and batch-mode analysis.

## Notes on `.streamlit/config.toml`
The file watcher is set to `fileWatcherType = "none"`. Streamlit's default watcher scans every
loaded module's `__path__` to detect source changes, and Transformers 5.x emits a deprecation
notice for each of its ~200 lazy-loaded vision-model submodules when that happens — flooding the
terminal with noise on every run. Disabling the watcher removes it. Trade-off: editing `dashboard.py`
while the app is running won't auto-reload — re-run `streamlit run dashboard.py` after changes,
or delete this file / set the value back to `"auto"` during active development.

## Installation & Usage
```bash
git clone https://github.com/tanu-dev0/mental_health_ai.git
cd mental_health_ai

python -m venv venv
.\venv\Scripts\activate        # on Windows
# source venv/bin/activate     # on macOS/Linux

pip install -r requirements.txt
streamlit run dashboard.py
```

To try the lightweight CLI version instead:
```bash
python app.py
```

## Running Tests
```bash
pip install -r requirements-dev.txt
pytest -v
```

## Project Structure
```
.
├── app.py                 # CLI demo entry point
├── dashboard.py            # Streamlit XAI dashboard (main entry point)
├── emotion_utils.py         # Shared, dependency-light helper logic
├── requirements.txt         # Pinned runtime dependencies
├── requirements-dev.txt      # Runtime + test dependencies
├── pytest.ini               # Test configuration
├── tests/
│   └── test_emotion_utils.py
└── LICENSE
```

## License
MIT — see [LICENSE](LICENSE).
