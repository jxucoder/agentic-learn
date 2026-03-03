import json
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

        if self.noise_cols_:
            X = X.drop(columns=self.noise_cols_, errors="ignore")

        for col in ["income", "age", "hours_worked", "distance_km", "satisfaction"]:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype(float)

        if "age" in X.columns:
            X["age_centered"] = X["age"] - 45.0
            X["age_centered_sq"] = (X["age"] - 45.0) ** 2

        if "hours_worked" in X.columns:
            X["log_hours"] = np.log1p(np.clip(X["hours_worked"], a_min=0.0, a_max=None))

        if "distance_km" in X.columns:
            X["log_distance"] = np.log1p(np.clip(X["distance_km"], a_min=0.0, a_max=None))
            X["distance_inv"] = 1.0 / (1.0 + np.clip(X["distance_km"], a_min=0.0, a_max=None))

        if {"log_hours", "log_distance"}.issubset(X.columns):
            X["log_hours_div_log_distance"] = X["log_hours"] / (X["log_distance"] + 1e-3)

        if "satisfaction" in X.columns:
            X["satisfaction_gt3"] = (X["satisfaction"] > 3).astype(float)
            X["satisfaction_centered"] = X["satisfaction"] - 3.0

        if {"income", "education"}.issubset(X.columns):
            edu_map = {"high_school": 1.0, "bachelors": 2.0, "masters": 3.0, "phd": 4.0}
            edu_ord = X["education"].map(edu_map).astype(float)
            X["income_x_education"] = X["income"] * edu_ord

        if {"region", "education"}.issubset(X.columns):
            X["region_education"] = X["region"].astype(str) + "__" + X["education"].astype(str)

        return X


def build_pipeline(X):
    fe_sample = FeatureEngineer().fit_transform(X.head(50))
    num_cols = fe_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = fe_sample.select_dtypes(exclude=[np.number]).columns.tolist()

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
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = LogisticRegression(
        penalty="l1",
        C=0.2,
        solver="liblinear",
        max_iter=5000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_f1_with_threshold_tuning(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    thresholds = np.linspace(0.2, 0.8, 121)
    best_f1 = 0.0
    for threshold in thresholds:
        preds = (oof_prob >= threshold).astype(int)
        score = f1_score(y, preds)
        if score > best_f1:
            best_f1 = float(score)
    return best_f1


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    pipeline = build_pipeline(X)
    metric = evaluate_f1_with_threshold_tuning(pipeline, X, y)

    result = {"metric": metric}
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
