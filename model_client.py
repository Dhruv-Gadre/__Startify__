import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to persisted model (training step will create this)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'funding_model.joblib')
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
META_PATH = os.path.join(os.path.dirname(__file__), 'models', 'funding_model_meta.json')
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'funding_pipeline.joblib')
PIPELINE_META_PATH = os.path.join(os.path.dirname(__file__), 'models', 'funding_pipeline_meta.json')

print("MODEL_PATH:", MODEL_PATH)
print("META_PATH:", META_PATH)
print("PIPELINE_PATH:", PIPELINE_PATH)
print("PIPELINE_META_PATH:", PIPELINE_META_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

_embed_model = None
_funding_model = None


def _load_models():
    global _embed_model, _funding_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    if _funding_model is None and os.path.exists(MODEL_PATH):
        _funding_model = joblib.load(MODEL_PATH)


def predict_funding(description, industry=None, city=None, founding_year=None):
    """
    Predict funding amount for a user's idea.

    Returns a dict: {"predicted_amount": float|None, "confidence": float (0-1), "error": str|None}
    If the persisted model is missing, returns predicted_amount=None and an error message.
    """
    try:
        _load_models()
        # Prefer the newer pipeline model if available
        if os.path.exists(PIPELINE_PATH) and os.path.exists(PIPELINE_META_PATH):
            try:
                pipeline = joblib.load(PIPELINE_PATH)
                import json
                with open(PIPELINE_META_PATH, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                # The pipeline expects a DataFrame input for its feature builder step.
                import pandas as pd
                input_df = pd.DataFrame([{
                    'Description': description,
                    'Starting Year': founding_year if founding_year is not None else 0,
                    'Founders Count': 0,
                    'No. of Employees': 0,
                    'City': city if city is not None else '',
                    'Industries': industry if industry is not None else ''
                }])

                # If pipeline exposes the underlying model with uncertainty, use it
                try:
                    model = pipeline.named_steps.get('model', None)
                except Exception:
                    model = None

                if model is not None and hasattr(model, 'predict_with_uncertainty'):
                    # Transform features using pipeline's feature builder
                    features = pipeline.named_steps['features'].transform(input_df)
                    mean_log, std_log = model.predict_with_uncertainty(features)
                    mean = mean_log[0]
                    std = std_log[0]
                    if meta.get('target_transform') == 'log1p':
                        # convert to dollar-space
                        pred_amount = float(np.expm1(mean))
                        # 95% interval
                        lower = float(np.expm1(mean - 1.96 * std))
                        upper = float(np.expm1(mean + 1.96 * std))
                    else:
                        pred_amount = float(mean)
                        lower = float(mean - 1.96 * std)
                        upper = float(mean + 1.96 * std)
                    # Provide a default heuristic confidence when uncertainty is available
                    # Here we return 0.6 (60%) as a baseline; this can be calibrated later.
                    return {"predicted_amount": pred_amount, "confidence": 0.6, "error": None, "lower": lower, "upper": upper}
                else:
                    # Fallback to simple predict
                    pred = pipeline.predict(input_df)[0]
                    if meta.get('target_transform') == 'log1p':
                        pred = np.expm1(pred)
                    return {"predicted_amount": float(pred), "confidence": 0.6, "error": None}
            except Exception as e:
                # fallback to older behavior if pipeline fails
                pass

        if _funding_model is None:
            return {"predicted_amount": None, "confidence": 0.0, "error": "No funding model found (train and save models/funding_model.joblib or funding_pipeline.joblib)"}

        # Create feature vector: embedding + simple numeric feature (founding_year)
        emb = _embed_model.encode([description])[0]
        # NOTE: Real training should include engineered categorical features and scaling.
        extra = np.array([founding_year if founding_year is not None else 0])
        features = np.hstack([emb, extra])

        pred = _funding_model.predict(features.reshape(1, -1))[0]

        # Check metadata to see if we need to inverse-transform (e.g., log1p)
        try:
            import json
            if os.path.exists(META_PATH):
                with open(META_PATH, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if meta.get('target_transform') == 'log1p':
                    # inverse transform
                    pred = np.expm1(pred)
        except Exception:
            # If anything goes wrong reading metadata, return raw pred
            pass

        # Confidence is a heuristic here (model-dependent). We'll return a placeholder 0.6
        return {"predicted_amount": float(pred), "confidence": 0.6, "error": None}

    except Exception as e:
        return {"predicted_amount": None, "confidence": 0.0, "error": str(e)}
