import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


DATA_PATH = Path('/Users/jiaruixu/work_space/agentic-learn/data/synth_reg.csv')
TARGET_COL = 'target'
RESULT_PATH = Path('result.json')
RANDOM_STATE = 42


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, include_noise=True):
        self.include_noise = include_noise
        self.base_num_cols = ['income', 'age', 'hours_worked', 'distance_km', 'satisfaction']
        self.noise_cols = ['noise_feature_a', 'noise_feature_b']
        self.cat_cols = ['region', 'education']

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_cols_ = list(self.base_num_cols)
        if self.include_noise:
            self.numeric_cols_ += [c for c in self.noise_cols if c in X.columns]

        self.medians_ = {
            c: pd.to_numeric(X[c], errors='coerce').median() for c in self.numeric_cols_
        }

        self.cat_maps_ = {}
        for c in self.cat_cols:
            vals = X[c].astype(str)
            uniques = pd.Index(vals.dropna().unique())
            self.cat_maps_[c] = {v: i for i, v in enumerate(uniques)}

        return self

    def transform(self, X):
        X = X.copy()

        out = pd.DataFrame(index=X.index)

        for c in self.numeric_cols_:
            out[c] = pd.to_numeric(X[c], errors='coerce').fillna(self.medians_[c])

        for c in self.cat_cols:
            mapping = self.cat_maps_[c]
            out[c] = X[c].astype(str).map(mapping).fillna(-1).astype(float)

        # Non-linear effects and interactions.
        out['income_x_hours'] = out['income'] * out['hours_worked']
        out['income_per_hour'] = out['income'] / out['hours_worked'].replace(0.0, np.nan)
        out['age_sq'] = out['age'] ** 2
        out['dist_sq'] = out['distance_km'] ** 2
        out['satisfaction_sq'] = out['satisfaction'] ** 2
        out['age_x_hours'] = out['age'] * out['hours_worked']

        out = out.replace([np.inf, -np.inf], np.nan)
        for c in out.columns:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].median())

        return out.to_numpy(dtype=float)


def build_pipeline(include_noise):
    model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_depth=5,
        learning_rate=0.04,
        max_iter=700,
        min_samples_leaf=12,
        l2_regularization=0.01,
    )
    return Pipeline([
        ('features', FeatureBuilder(include_noise=include_noise)),
        ('model', model),
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].to_numpy()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    score_with_noise = cross_val_score(
        build_pipeline(include_noise=True), X, y, scoring='r2', cv=cv
    ).mean()
    score_without_noise = cross_val_score(
        build_pipeline(include_noise=False), X, y, scoring='r2', cv=cv
    ).mean()

    metric = float(max(score_with_noise, score_without_noise))
    payload = {'metric': metric}

    RESULT_PATH.write_text(json.dumps(payload))
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
