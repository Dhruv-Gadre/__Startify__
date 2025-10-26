# Startify — AI Startup Evaluation

> Startify helps founders quickly evaluate startup ideas using similarity search, an LLM-powered "copycat" analysis, and a funding-prediction model. It includes a Streamlit prototype UI, a FastAPI backend used by a static web front-end, and training scripts to build a funding model from sample startup data.

## What this repository contains

- `app.py` — Streamlit demo app (interactive local UI).
- `main.py` — FastAPI backend that exposes a POST `/analyze` endpoint used by the web UI.
- `index.html`, `evaluate.html`, `script.js`, `styles.css` — simple static website that calls the FastAPI backend.
- `model_client.py` — runtime wrapper that loads saved funding prediction models and returns structured predictions.
- `model_parts.py` — feature builder and ensemble estimator used during model training.
- `train_funding_model.py` — training script that produces a pipeline saved under `models/`.
- `ollama_client.py` — helper that queries a local Ollama LLM to perform similarity / "copycat" analysis.
- `Startups1.csv`, `startup_funding.csv`, `startup_failure_prediction.csv` — sample data files used by the app and training scripts.
- `smoke_predict.py` — tiny smoke test script that calls `model_client.predict_funding` on sample ideas.
- `requirements.txt` — Python dependencies for the project.

## High-level flow

1. A user submits a startup idea.
2. The system finds the top-k most similar startups using SentenceTransformers + FAISS.
3. The top match is analyzed using a local LLM (Ollama) to produce a similarity score and a simple "is_copy" verdict.
4. A funding model returns a predicted funding amount and optional uncertainty interval.

## Quickstart (local, development)

Prerequisites

- Python 3.9+ (3.10/3.11 recommended)
- Git (optional)
- (Optional) Ollama installed and a compatible model downloaded locally if you want the LLM analysis to work.

Install Python dependencies (PowerShell example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Notes on `faiss-cpu` and SentenceTransformers:
- `faiss-cpu` is listed in `requirements.txt`. On some Windows environments installing `faiss-cpu` via pip can be tricky — if you hit wheel/platform issues, use conda or follow the FAISS installation instructions for Windows.

Running the Streamlit demo UI

```powershell
streamlit run app.py
```

Open the app in your browser (Streamlit usually opens at http://localhost:8501).

Running the FastAPI backend (used by the static web UI)

```powershell
# From the repository root
uvicorn main:app --reload --port 8000
```

Visit `evaluate.html` in the repo (open in a browser) — this static page calls the backend at `http://127.0.0.1:8000/analyze`.

Training the funding model

1. Ensure `Startups1.csv` is present and contains examples with `Funding Amount in $` (the training script filters to positive funding rows).
2. Run the trainer to produce `models/funding_pipeline.joblib` and metadata:

```powershell
python train_funding_model.py
```

If training completes successfully, a pipeline and `funding_pipeline_meta.json` will be saved to the `models/` folder. The FastAPI and frontend will use these artifacts via `model_client.py`.

API usage

POST /analyze

Request body (JSON):

```json
{
  "user_idea": "My idea description...",
  "user_city": "City name",
  "founding_year": 2025
}
```

Response (summary):

```json
{
  "profile": { ... },
  "ai_analysis": { "top_match_name": "...", "score": 78, "is_copy": false, "reasoning": "..." },
  "similar_startups": [ ... ],
  "funding_analysis": { ... },
  "funding_prediction": { "predicted_amount": 50000, "confidence": 0.6, ... }
}
```

Project structure quick reference

- `app.py` — run a local Streamlit UI.
- `main.py` — run a FastAPI backend (used by `evaluate.html` + `script.js`).
- `model_client.py` — model serving logic; it looks for models in `models/`.
- `model_parts.py`, `train_funding_model.py` — training pipeline.
- `ollama_client.py` — small helper that calls the local Ollama HTTP endpoint and returns JSON.
