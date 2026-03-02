import json
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


DATA_PATH = "/Users/jiaruixu/work_space/agentic-learn/data/titanic.csv"
TARGET_COL = "survived"
LEAKAGE_COLS = ["boat", "body", "home.dest"]


def extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"

    match = re.search(r",\s*([^\.]+)\.", name)
    if not match:
        return "Unknown"

    title = match.group(1).strip()
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "the Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    title = title_map.get(title, title)

    if title not in {"Mr", "Mrs", "Miss", "Master"}:
        return "Rare"
    return title


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x.drop(columns=LEAKAGE_COLS, errors="ignore")

    x["title"] = x["name"].apply(extract_title)
    x["family_size"] = x["sibsp"].fillna(0) + x["parch"].fillna(0) + 1
    x["is_alone"] = (x["family_size"] == 1).astype(int)
    x["cabin_group"] = x["cabin"].fillna("U").astype(str).str[0].str.upper().replace({"T": "A"})

    ticket = x["ticket"].fillna("UNK").astype(str)
    x["ticket_prefix"] = (
        ticket.str.upper().str.replace(r"[^A-Z./ ]", "", regex=True).str.strip().replace("", "NONE")
    )
    x["ticket_group_size"] = ticket.map(ticket.value_counts()).fillna(1)

    x["fare_per_person"] = x["fare"] / x["family_size"].replace(0, 1)
    x["fare_per_person"] = x["fare_per_person"].replace([np.inf, -np.inf], np.nan)
    x["name_len"] = x["name"].fillna("").str.len()
    x["age_missing"] = x["age"].isna().astype(int)
    x["class_sex"] = x["pclass"].astype(str) + "_" + x["sex"].astype(str)

    surname = x["name"].str.extract(r"^([^,]+),", expand=False).fillna("Unknown")
    x["surname_group_size"] = surname.map(surname.value_counts()).fillna(1)

    return x


def build_pipeline(x_train: pd.DataFrame) -> Pipeline:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    return Pipeline([("preprocess", preprocess), ("model", model)])


def choose_threshold(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_oof_proba = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    thresholds = np.linspace(0.20, 0.80, 121)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (train_oof_proba >= threshold).astype(int)
        score = f1_score(y_train, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = df[TARGET_COL].astype(int)
    x = df.drop(columns=[TARGET_COL])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline(x_train)
    threshold = choose_threshold(pipeline, x_train, y_train)

    pipeline.fit(x_train, y_train)
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    metric = float(f1_score(y_test, test_pred))

    payload = {"metric": round(metric, 6)}
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
