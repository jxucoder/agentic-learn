import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "/Users/jiaruixu/work_space/agentic-learn/data/synth_clf.csv"
TARGET_COL = "target"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer non-linear and interaction features from mixed-type inputs."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in ["income", "age", "hours_worked", "distance_km", "satisfaction"]:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype(int)

        if "income" in X.columns:
            X["log_income"] = np.log1p(np.clip(X["income"], a_min=0, a_max=None))

        if "distance_km" in X.columns:
            X["log_distance"] = np.log1p(np.clip(X["distance_km"], a_min=0, a_max=None))

        if "hours_worked" in X.columns:
            X["log_hours"] = np.log1p(np.clip(X["hours_worked"], a_min=0, a_max=None))

        if "age" in X.columns:
            X["age_sq"] = X["age"] ** 2
            X["age_abs_45"] = np.abs(X["age"] - 45.0)

        if {"hours_worked", "distance_km"}.issubset(X.columns):
            hours = np.clip(X["hours_worked"], a_min=1e-6, a_max=None)
            dist = np.clip(X["distance_km"], a_min=1e-6, a_max=None)
            X["log_hours_over_log_distance"] = np.log1p(hours) / np.log1p(dist)
            X["distance_x_hours"] = X["distance_km"] * X["hours_worked"]

        if "satisfaction" in X.columns:
            X["sat_gt3"] = (X["satisfaction"] > 3).astype(float)
            X["sat_le2"] = (X["satisfaction"] <= 2).astype(float)

        if {"income", "education"}.issubset(X.columns):
            edu_map = {"high_school": 1, "bachelors": 2, "masters": 3, "phd": 4}
            edu_ord = X["education"].map(edu_map)
            X["income_x_edu_ord"] = X["income"] * edu_ord

        if {"region", "education"}.issubset(X.columns):
            X["region_education"] = X["region"].astype(str) + "__" + X["education"].astype(str)

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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    base = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocess", preprocess),
            # Additional in-fold suppression of low-information (noise-like) features.
            ("feature_select", SelectPercentile(score_func=mutual_info_classif, percentile=70)),
        ]
    )

    stack = StackingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    C=2.5,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=600,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(48, 24),
                    alpha=3e-4,
                    learning_rate_init=2e-3,
                    max_iter=1200,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ],
        final_estimator=LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=5000,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
        passthrough=False,
    )

    return Pipeline(
        steps=[
            ("base", base),
            ("model", stack),
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
