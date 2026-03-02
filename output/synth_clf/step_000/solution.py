import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


DATA_PATH = "/Users/jiaruixu/work_space/agentic-learn/data/synth_clf.csv"
TARGET_COL = "target"


class FeatureEngineerAndNoiseFilter(BaseEstimator, TransformerMixin):
    """Drop low-signal noise columns and add non-linear / interaction features."""

    def __init__(self, mi_threshold=0.001, random_state=42):
        self.mi_threshold = mi_threshold
        self.random_state = random_state
        self.noise_cols_to_drop_ = []

    def fit(self, X, y=None):
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if y is None or not numeric_cols:
            self.noise_cols_to_drop_ = []
            return self

        X_num = X[numeric_cols].copy()
        X_num = X_num.fillna(X_num.median(numeric_only=True))
        mi = mutual_info_classif(X_num, y, random_state=self.random_state)

        # Keep signal-bearing numeric columns; only auto-drop explicit noise columns.
        self.noise_cols_to_drop_ = [
            col
            for col, score in zip(numeric_cols, mi)
            if col.startswith("noise_") and score <= self.mi_threshold
        ]
        return self

    def transform(self, X):
        X = X.copy()

        if self.noise_cols_to_drop_:
            X = X.drop(columns=[c for c in self.noise_cols_to_drop_ if c in X.columns])

        # Interaction and non-linear transforms.
        if {"income", "hours_worked"}.issubset(X.columns):
            X["income_x_hours"] = X["income"] * X["hours_worked"]
            X["income_per_hour"] = X["income"] / (X["hours_worked"].abs() + 1.0)

        if "age" in X.columns:
            X["age_sq"] = X["age"] ** 2

        if "distance_km" in X.columns:
            X["distance_sq"] = X["distance_km"] ** 2

        if {"satisfaction", "income"}.issubset(X.columns):
            X["satisfaction_x_income"] = X["satisfaction"] * X["income"]

        if {"distance_km", "hours_worked"}.issubset(X.columns):
            X["distance_x_hours"] = X["distance_km"] * X["hours_worked"]

        return X


def build_pipeline(X, y):
    fe = FeatureEngineerAndNoiseFilter(mi_threshold=0.001, random_state=42)
    X_fe = fe.fit_transform(X, y)

    numeric_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    clf = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_depth=5,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", fe),
            ("preprocess", preprocess),
            ("model", clf),
        ]
    )


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    pipeline = build_pipeline(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )
    metric = float(scores.mean())

    payload = {"metric": metric}
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
