import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


DATA_PATH = Path('/Users/jiaruixu/work_space/agentic-learn/data/synth_reg.csv')
TARGET_COL = 'target'
RESULT_PATH = Path('result.json')
RANDOM_STATE = 42


class StructuredFeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds non-linear and interaction features aligned with discovered data structure."""

    def __init__(self, include_noise=False):
        self.include_noise = include_noise
        self.base_numeric_cols = ['income', 'age', 'hours_worked', 'distance_km', 'satisfaction']
        self.cat_cols = ['region', 'education']
        self.noise_cols = ['noise_feature_a', 'noise_feature_b']

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_cols_ = list(self.base_numeric_cols)
        if self.include_noise:
            self.numeric_cols_ += [c for c in self.noise_cols if c in X.columns]

        self.numeric_medians_ = {
            c: pd.to_numeric(X[c], errors='coerce').median() for c in self.numeric_cols_
        }

        self.cat_levels_ = {}
        for c in self.cat_cols:
            vals = X[c].astype(str).fillna('__MISSING__')
            self.cat_levels_[c] = list(pd.Index(vals.unique()))

        return self

    def transform(self, X):
        X = X.copy()

        for c in self.numeric_cols_:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(self.numeric_medians_[c])
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna('__MISSING__')

        out = pd.DataFrame(index=X.index)

        income = X['income']
        age = X['age']
        hours = X['hours_worked']
        distance = X['distance_km']
        satisfaction = X['satisfaction']

        # Base informative features.
        out['income'] = income
        out['age'] = age
        out['hours_worked'] = hours
        out['distance_km'] = distance
        out['satisfaction'] = satisfaction

        # Non-linear transformations and interactions.
        out['age_center_sq'] = (age - 45.0) ** 2
        out['log_ratio_hours_distance'] = np.log1p(np.clip(hours, 0.0, None)) / (
            np.log1p(np.clip(distance, 0.0, None)) + 1e-6
        )
        out['satisfaction_high'] = (satisfaction > 3.0).astype(float)
        out['income_log'] = np.log1p(np.clip(income, 0.0, None))
        out['distance_log'] = np.log1p(np.clip(distance, 0.0, None))
        out['income_x_hours'] = income * hours
        out['hours_x_distance'] = hours * distance

        if self.include_noise:
            for c in self.noise_cols:
                if c in X.columns:
                    out[c] = X[c]

        # Stable one-hot encoding for categoricals.
        for c in self.cat_cols:
            vals = X[c].astype(str)
            for level in self.cat_levels_[c]:
                out[f'{c}__{level}'] = (vals == level).astype(float)

        # Captures multiplicative effect: income × education.
        for level in self.cat_levels_['education']:
            edu_col = f'education__{level}'
            out[f'income_x_{edu_col}'] = income * out[edu_col]

        out = out.replace([np.inf, -np.inf], np.nan)
        for c in out.columns:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].median())

        return out.to_numpy(dtype=float)


def build_pipeline(include_noise):
    return Pipeline([
        ('features', StructuredFeatureBuilder(include_noise=include_noise)),
        ('model', RidgeCV(alphas=np.logspace(-4, 4, 41))),
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].to_numpy()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    score_without_noise = cross_val_score(
        build_pipeline(include_noise=False), X, y, scoring='r2', cv=cv
    ).mean()
    score_with_noise = cross_val_score(
        build_pipeline(include_noise=True), X, y, scoring='r2', cv=cv
    ).mean()

    metric = float(max(score_without_noise, score_with_noise))
    payload = {'metric': metric}

    RESULT_PATH.write_text(json.dumps(payload))
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
