import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "/Users/jiaruixu/work_space/agentic-learn/data/synth_clf.csv"
TARGET_COL = "target"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.noise_cols_ = None

    def fit(self, X, y=None):
        self.noise_cols_ = sorted([c for c in X.columns if "noise" in c.lower()])
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.noise_cols_, errors="ignore")

        for col in ["income", "age", "hours_worked", "distance_km", "satisfaction"]:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype(float)

        if "income" in X.columns:
            X["log_income"] = np.log1p(np.clip(X["income"], a_min=0.0, a_max=None))
            X["income_sq"] = X["income"] ** 2

        if "age" in X.columns:
            X["age_centered"] = X["age"] - 45.0
            X["age_centered_sq"] = (X["age"] - 45.0) ** 2

        if "hours_worked" in X.columns:
            X["log_hours"] = np.log1p(np.clip(X["hours_worked"], a_min=0.0, a_max=None))
            X["hours_sq"] = X["hours_worked"] ** 2

        if "distance_km" in X.columns:
            dist = np.clip(X["distance_km"], a_min=0.0, a_max=None)
            X["log_distance"] = np.log1p(dist)
            X["distance_inv"] = 1.0 / (1.0 + dist)
            X["distance_sq"] = X["distance_km"] ** 2

        if {"hours_worked", "distance_km"}.issubset(X.columns):
            X["hours_div_distance"] = X["hours_worked"] / (1.0 + X["distance_km"])
            X["hours_x_distance"] = X["hours_worked"] * X["distance_km"]

        if "satisfaction" in X.columns:
            X["satisfaction_gt3"] = (X["satisfaction"] > 3).astype(float)
            X["satisfaction_centered"] = X["satisfaction"] - 3.0

        if {"age", "satisfaction"}.issubset(X.columns):
            X["age_x_satisfaction"] = X["age"] * X["satisfaction"]

        if {"income", "education"}.issubset(X.columns):
            edu_map = {"high_school": 1.0, "bachelors": 2.0, "masters": 3.0, "phd": 4.0}
            edu_ord = X["education"].map(edu_map).astype(float)
            X["edu_ord"] = edu_ord
            X["income_x_education"] = X["income"] * edu_ord

        if {"region", "education"}.issubset(X.columns):
            X["region_education"] = X["region"].astype(str) + "__" + X["education"].astype(str)

        if {"region", "hours_worked"}.issubset(X.columns):
            buckets = pd.cut(
                X["hours_worked"],
                bins=[-np.inf, 30.0, 40.0, 50.0, np.inf],
                labels=["lt30", "30_40", "40_50", "gt50"],
            ).astype(str)
            X["region_hours_bucket"] = X["region"].astype(str) + "__" + buckets

        return X


def build_pipeline(X):
    fe = FeatureEngineer()
    transformed = fe.fit_transform(X)

    numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = transformed.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = LogisticRegression(
        penalty="l2",
        C=2.0,
        solver="lbfgs",
        max_iter=6000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_f1_oof_threshold(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    best_f1 = 0.0
    for threshold in np.linspace(0.2, 0.8, 121):
        preds = (oof_prob >= threshold).astype(int)
        score = f1_score(y, preds)
        if score > best_f1:
            best_f1 = float(score)
    return best_f1


def main():
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    warnings.filterwarnings("ignore")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    pipeline = build_pipeline(X)
    metric = evaluate_f1_oof_threshold(pipeline, X, y)

    result = {"metric": metric}
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
