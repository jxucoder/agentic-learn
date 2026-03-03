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


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "income" in X.columns:
            X["income_log"] = np.log1p(X["income"].clip(lower=0))
            X["income_sq"] = X["income"] ** 2
        if "distance_km" in X.columns:
            X["distance_log"] = np.log1p(X["distance_km"].clip(lower=0))
            X["distance_sq"] = X["distance_km"] ** 2
            X["distance_inv"] = 1.0 / (1.0 + X["distance_km"])
        if "age" in X.columns:
            X["age_sq"] = X["age"] ** 2
        if "hours_worked" in X.columns:
            X["hours_sq"] = X["hours_worked"] ** 2

        if {"income", "distance_km"}.issubset(X.columns):
            X["income_x_distance"] = X["income"] * X["distance_km"]
        if {"satisfaction", "hours_worked"}.issubset(X.columns):
            X["sat_x_hours"] = X["satisfaction"] * X["hours_worked"]
        if {"income", "hours_worked"}.issubset(X.columns):
            X["income_per_hour"] = X["income"] / (1.0 + X["hours_worked"])
        if {"region", "education"}.issubset(X.columns):
            X["region_education"] = X["region"].astype(str) + "__" + X["education"].astype(str)

        return X


def identify_noise_columns(X, y):
    """Drop only explicit noise candidates that also show near-zero MI."""
    candidates = [c for c in X.columns if "noise" in c.lower()]
    if not candidates:
        return []

    encoded = X.copy()
    cat_cols = encoded.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        encoded[col] = encoded[col].astype("category").cat.codes
    for col in encoded.columns:
        if encoded[col].isna().any():
            encoded[col] = encoded[col].fillna(encoded[col].median())

    mi_scores = mutual_info_classif(encoded, y, random_state=42)
    mi_map = dict(zip(encoded.columns, mi_scores))
    return [c for c in candidates if mi_map.get(c, 0.0) <= 0.005]


def build_pipeline(X, y):
    noise_cols = identify_noise_columns(X, y)
    X_model = X.drop(columns=noise_cols) if noise_cols else X

    fe = FeatureEngineer()
    sample_fe = fe.fit_transform(X_model.head(5))
    num_cols = sample_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = sample_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
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
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", fe),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline, X_model


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    pipeline, X_model = build_pipeline(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline,
        X_model,
        y,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )
    metric = float(np.mean(scores))

    print(json.dumps({"metric": metric}))


if __name__ == "__main__":
    main()
