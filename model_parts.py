import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, embed_model_name='all-MiniLM-L6-v2'):
        self.embed_model_name = embed_model_name
        self.embed_model = None
        self.city_counts = {}
        self.industry_counts = {}

    def fit(self, X_df, y=None):
        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.city_counts = X_df['City'].fillna('___missing___').value_counts().to_dict()
        self.industry_counts = X_df['Industries'].fillna('___missing___').value_counts().to_dict()
        return self

    def transform(self, X_df):
        descs = X_df['Description'].fillna('').tolist()
        emb = self.embed_model.encode(descs, show_progress_bar=False)
        years = X_df['Starting Year'].fillna(0).astype(float).values.reshape(-1, 1)
        founders = X_df.get('Founders Count', pd.Series([0]*len(X_df))).fillna(0).astype(float).values.reshape(-1, 1)
        employees = X_df.get('No. of Employees', pd.Series([0]*len(X_df))).fillna(0).astype(float).values.reshape(-1, 1)
        city_feat = X_df['City'].fillna('___missing___').map(self.city_counts).fillna(0).astype(float).values.reshape(-1, 1)
        industry_feat = X_df['Industries'].fillna('___missing___').map(self.industry_counts).fillna(0).astype(float).values.reshape(-1, 1)
        X_feats = np.hstack([emb, years, founders, employees, city_feat, industry_feat])
        return X_feats


class EnsembleEstimator(BaseEstimator):
    def __init__(self, base_estimator=None, n_members=5):
        self.base_estimator = base_estimator if base_estimator is not None else RandomForestRegressor()
        self.n_members = n_members
        self.members_ = []

    def fit(self, X, y):
        self.members_ = []
        for i in range(self.n_members):
            member = clone(self.base_estimator)
            try:
                setattr(member, 'random_state', int(42 + i))
            except Exception:
                pass
            member.fit(X, y)
            self.members_.append(member)
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.members_])
        return preds.mean(axis=1)

    def predict_with_uncertainty(self, X):
        preds = np.column_stack([m.predict(X) for m in self.members_])
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        return mean, std
