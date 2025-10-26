# Startify — AI Startup Evaluation

> Startify helps founders quickly evaluate startup ideas using similarity search, an LLM-powered "copycat" analysis, and a funding-prediction model. It includes a Streamlit prototype UI, a FastAPI backend used by a static web front-end, and training scripts to build a funding model from sample startup data.

![landing page](https://github.com/user-attachments/assets/09ab3db0-cd7d-4fe4-82f9-c6cb2fb52fdd)


## High-level flow

1. A user submits a startup idea.
![1](https://github.com/user-attachments/assets/d28836a5-1bae-43ff-b763-1fc95a45f52f)

2. The system finds the top-k most similar startups using SentenceTransformers + FAISS.
![2](https://github.com/user-attachments/assets/f31d2ef6-dae4-4f71-83ad-0ec84c5a1ae8)


3. The top match is analyzed using a local LLM (Ollama) to produce a similarity score and a simple "is_copy" verdict.
![3](https://github.com/user-attachments/assets/34912487-8db4-4577-8f64-8ba7f1e6fc3f)


4. A funding model returns a predicted funding amount and optional uncertainty interval.
![4](https://github.com/user-attachments/assets/4b11282b-a065-4b02-8004-06c2dcaffd90)

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
