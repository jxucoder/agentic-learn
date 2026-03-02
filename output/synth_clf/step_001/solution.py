import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "/Users/jiaruixu/work_space/agentic-learn/data/synth_clf.csv"
TARGET_COL = "target"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add non-linear terms, interactions, and robust missingness indicators."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "income" in X.columns:
            X["income_missing"] = X["income"].isna().astype(int)
            X["log_income"] = np.log1p(np.clip(X["income"], a_min=0, a_max=None))
            X["income_sq"] = X["income"] ** 2

        if "distance_km" in X.columns:
            X["distance_missing"] = X["distance_km"].isna().astype(int)
            X["log_distance"] = np.log1p(np.clip(X["distance_km"], a_min=0, a_max=None))
            X["distance_sq"] = X["distance_km"] ** 2

        if "age" in X.columns:
            X["age_sq"] = X["age"] ** 2
            X["age_bucket"] = pd.cut(
                X["age"],
                bins=[0, 25, 35, 45, 60, 200],
                labels=["18_25", "26_35", "36_45", "46_60", "60_plus"],
            ).astype(object)

        if "hours_worked" in X.columns:
            X["hours_sq"] = X["hours_worked"] ** 2

        if {"income", "hours_worked"}.issubset(X.columns):
            X["income_x_hours"] = X["income"] * X["hours_worked"]
            X["income_per_hour"] = X["income"] / (X["hours_worked"].abs() + 1.0)

        if {"distance_km", "hours_worked"}.issubset(X.columns):
            X["distance_x_hours"] = X["distance_km"] * X["hours_worked"]

        if {"age", "income"}.issubset(X.columns):
            X["age_x_income"] = X["age"] * X["income"]

        if {"region", "education"}.issubset(X.columns):
            X["region_education"] = X["region"].astype(str) + "__" + X["education"].astype(str)

        if "satisfaction" in X.columns:
            X["satisfaction_cat"] = X["satisfaction"].astype("Int64").astype(str)

        return X


def build_pipeline(X):
    sample = FeatureEngineer().fit_transform(X)
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in sample.columns if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=0.01,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocess", preprocess),
            (
                "feature_select",
                SelectPercentile(score_func=mutual_info_classif, percentile=65),
            ),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    C=4.0,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    pipeline = build_pipeline(X)
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
