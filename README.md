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

## AI / ML pipeline

This project contains three collaborating ML components: (A) the similarity search pipeline (embeddings + FAISS), (B) an LLM-based similarity/"copycat" analysis, and (C) a funding prediction pipeline. Together they power the UI and API.

1) Data and inputs
- Primary dataset: `Startups1.csv` (used by the app and the trainer). Columns used include `Company`, `Description`, `Industries`, `City`, `Funding Amount in $`, `Funding Round`, and optional features like `Starting Year`, `Founders Count`, and `No. of Employees`.
- The training script (`train_funding_model.py`) filters to rows with positive funding for regression stability. Ensure the CSV is cleaned of malformed currency strings (the repo strips `$` and `,` during loading).

2) Preprocessing
- The app and training code perform lightweight cleaning in Python: fill missing descriptions/industries, normalize the funding column to numeric, and compute an `id` index used by FAISS.
- The training `FeatureBuilder` (in `model_parts.py`) is the canonical feature pipeline: it builds sentence embeddings for descriptions, encodes numeric features (starting year, founders, employees) and creates simple aggregated features (city and industry counts).

3) Embeddings + FAISS (similarity search)
- Embeddings: `sentence-transformers` model `all-MiniLM-L6-v2` (see `app.py` and `main.py`) is used to convert startup descriptions into dense vectors.
- Indexing: FAISS (`faiss-cpu`) builds an L2 index (`IndexFlatL2`) wrapped in an ID map so results map back to original rows. The app builds the index at startup (cached in memory in Streamlit; in FastAPI it's created during the `startup` event).
- Query: At inference, the user idea is encoded and top-k nearest neighbors are retrieved with `index.search(query_vector, k)`. Results are then re-ordered to match the returned neighbor IDs.

4) LLM "copycat" analysis (Ollama)
- The repo includes `ollama_client.py` which calls a local Ollama server at `http://127.0.0.1:11434/api/generate`. The model name is specified by `MODEL_NAME` in that file.
- Flow: the app sends a short prompt containing the user idea and the top match description and asks the LLM to return a small JSON object with `similarity_score` (0–100), `is_copy` (bool), and `reasoning` (string). The app expects strictly JSON output — in practice it's useful to add additional robustness parsing because large models sometimes include extra text.
- Note: If Ollama is not running, the Streamlit app will show a warning and the FastAPI endpoint returns 503. Install and run Ollama locally and download a model to enable this feature.

5) Funding model (training and inference)
- Training: `train_funding_model.py` composes a Pipeline: `('features', FeatureBuilder())` → `('model', EnsembleEstimator(...))`. The script transforms the target using `log1p` (`y = np.log1p(funding)`) before training the ensemble.
- EnsembleEstimator: trains multiple RandomForest members and exposes `predict` (mean) and `predict_with_uncertainty` (mean, std) so the code can report a 95% approximate interval (mean ± 1.96 * std) in log-space and invert with `expm1`.
- Persistence: the training script saves `models/funding_pipeline.joblib` (pipeline) and `models/funding_pipeline_meta.json` describing target transforms and feature order. `model_client.py` looks for these artifacts and falls back to an older `funding_model.joblib` if present.
- Inference: `model_client.predict_funding()` wraps model loading and converts an input idea + metadata into features. If the pipeline exposes `predict_with_uncertainty`, the client returns `predicted_amount`, `confidence` (heuristic), and `lower`/`upper` bounds. If models are missing, the client returns a clear error message that the frontend displays.

6) Evaluation & metrics
- The training script prints a validation RMSE on the log1p target. Use this RMSE to track improvements when changing features or model families.
- Recommended additional metrics: median absolute error in dollars (after inverse-transform), calibration of predicted intervals, and coverage of the 95% interval on a validation set.

7) Operational notes, edge cases & scaling
- Data quality: regression relies on enough funded examples. The training script exits early if there are too few positive examples.
- FAISS persistence: currently the FAISS index is built at startup and kept in memory. For larger datasets, persist and load the index to avoid recomputing embeddings on every start. FAISS provides `write_index` / `read_index` utilities.
- Embedding performance: encoding many descriptions is CPU-bound; use a GPU for faster embeddings (install `sentence-transformers` with PyTorch/CUDA) or perform offline embedding precomputation.
- LLM responses: guard against non-JSON responses. Consider a retry + cleanup pipeline or use a wrapper that extracts the first JSON blob from text.
- Concurrency: the FastAPI app loads models at startup and serves requests; in high-load setups move heavy work (embedding + index search) to worker processes or a lightweight cache layer.

8) Reproducibility and versioning
- Pin `sentence-transformers` and FAISS versions in `requirements.txt` or provide a `environment.yml` for conda (recommended on Windows for FAISS).
- Keep model artifacts under `models/` and tag them with training metadata (git commit, dataset hash, training params). Add `funding_pipeline_meta.json` fields such as `git_commit`, `train_date`, and `train_rows` for traceability.

Files to inspect for ML logic
- `train_funding_model.py` — training loop and model save.
- `model_parts.py` — `FeatureBuilder` and `EnsembleEstimator` implementations.
- `model_client.py` — loading and inference wrapper used by the API/frontend.
- `app.py` and `main.py` — example usage of embeddings, FAISS, and LLM integration.



