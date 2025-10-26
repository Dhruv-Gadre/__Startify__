"""
Simple training script for a funding prediction model.
It trains a RandomForestRegressor on embeddings + founding year.
This is a minimal example extend feature engineering for production.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
import json

# Ensure repo root is on sys.path so imports like `model_parts` work when running from scripts/
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_parts import FeatureBuilder, EnsembleEstimator

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(ROOT, 'Startups1.csv')
OUT_MODEL_DIR = os.path.join(ROOT, 'models')
OUT_MODEL_PATH = os.path.join(OUT_MODEL_DIR, 'funding_model.joblib')

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    # Basic cleaning used by the app
    df['Description'] = df['Description'].fillna('')
    df['Funding Amount in $'] = df['Funding Amount in $'].astype(str).str.replace(r'[$,]', '', regex=True)
    df['Funding Amount in $'] = pd.to_numeric(df['Funding Amount in $'], errors='coerce').fillna(0)
    df['Starting Year'] = pd.to_numeric(df.get('Starting Year', pd.Series([0]*len(df))), errors='coerce').fillna(0)
    # Parse founders count if present
    if 'Founders Count' in df.columns:
        df['Founders Count'] = pd.to_numeric(df['Founders Count'], errors='coerce').fillna(0)
    else:
        df['Founders Count'] = 0
    # Parse employees column heuristically
    if 'No. of Employees' in df.columns:
        df['No. of Employees'] = df['No. of Employees'].astype(str).str.extract(r'(\d+)', expand=False)
        df['No. of Employees'] = pd.to_numeric(df['No. of Employees'], errors='coerce').fillna(0)
    else:
        df['No. of Employees'] = 0
    return df

def main():
    df = load_and_prepare()
    # Use only rows with positive funding to help regression stability
    df_train = df[df['Funding Amount in $'] > 0].copy()
    if df_train.shape[0] < 10:
        raise SystemExit('Not enough funded examples to train a model.')



    feature_builder = FeatureBuilder()
    X_df = df_train[['Description', 'Starting Year', 'Founders Count', 'No. of Employees', 'City', 'Industries']]
    y_raw = df_train['Funding Amount in $'].values
    y = np.log1p(y_raw)

    # Build pipeline and fit
    pipeline = Pipeline([
        ('features', feature_builder),
        ('model', EnsembleEstimator(base_estimator=RandomForestRegressor(n_estimators=200), n_members=5))
    ])

    X_train_df, X_val_df, y_train, y_val = train_test_split(X_df, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train_df, y_train)

    # Evaluate
    model = pipeline.named_steps['model']
    feats_val = pipeline.named_steps['features'].transform(X_val_df)
    mean_log, std_log = model.predict_with_uncertainty(feats_val)
    preds = mean_log
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f'Validation RMSE (log1p target): {rmse:.4f}')

    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    pipeline_path = os.path.join(OUT_MODEL_DIR, 'funding_pipeline.joblib')
    joblib.dump(pipeline, pipeline_path)

    meta = {
        'target_transform': 'log1p',
        'embed_model': 'all-MiniLM-L6-v2',
        'feature_order': ['embedding', 'starting_year', 'founders_count', 'employees_count', 'city_count', 'industry_count']
    }
    meta_path = os.path.join(OUT_MODEL_DIR, 'funding_pipeline_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f)

    print('Saved model pipeline to', pipeline_path)
    print('Saved metadata to', meta_path)

if __name__ == '__main__':
    main()
