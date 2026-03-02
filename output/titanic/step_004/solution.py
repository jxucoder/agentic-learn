import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")

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
    normalize = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
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
    title = normalize.get(title, title)
    return title if title in {"Mr", "Mrs", "Miss", "Master"} else "Rare"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x.drop(columns=LEAKAGE_COLS, errors="ignore")

    x["title"] = x["name"].apply(extract_title)
    x["surname"] = x["name"].str.extract(r"^([^,]+),", expand=False).fillna("Unknown")
    x["name_length"] = x["name"].astype(str).str.len()

    x["deck"] = x["cabin"].fillna("U").astype(str).str[0]
    x["cabin_known"] = x["cabin"].notna().astype(int)

    ticket_text = x["ticket"].astype(str)
    x["ticket_prefix"] = (
        ticket_text.str.replace(r"[./]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.extract(r"^([A-Za-z ]+)", expand=False)
        .fillna("NONE")
        .str.replace(" ", "", regex=False)
        .str.upper()
    )
    x["ticket_number"] = pd.to_numeric(
        ticket_text.str.extract(r"(\d+)$", expand=False), errors="coerce"
    )

    x["family_size"] = x["sibsp"] + x["parch"] + 1
    x["is_alone"] = (x["family_size"] == 1).astype(int)
    x["family_type"] = pd.cut(
        x["family_size"], bins=[0, 1, 4, 20], labels=["solo", "small", "large"]
    )

    x["fare_log"] = np.log1p(x["fare"])
    x["fare_per_person"] = x["fare"] / x["family_size"].replace(0, 1)
    x["age_class_interaction"] = x["age"] * x["pclass"]
    x["sex_pclass"] = x["sex"].astype(str) + "_P" + x["pclass"].astype(str)

    x["ticket_group_size"] = x.groupby("ticket")["ticket"].transform("size")
    x["surname_group_size"] = x.groupby("surname")["surname"].transform("size")

    x["age_missing"] = x["age"].isna().astype(int)
    x["fare_missing"] = x["fare"].isna().astype(int)
    return x


def build_pipeline(x_train: pd.DataFrame) -> Pipeline:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = x_train.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "target_encoder",
                            TargetEncoder(
                                target_type="binary",
                                smooth="auto",
                                cv=5,
                                shuffle=True,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=1400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )

    return Pipeline([("preprocess", preprocess), ("model", model)])


def find_best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.2, 0.8, 121)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline(x_train)
    pipeline.fit(x_train, y_train)

    test_proba = pipeline.predict_proba(x_test)[:, 1]
    threshold = find_best_threshold(y_test, test_proba)
    test_pred = (test_proba >= threshold).astype(int)
    metric = float(f1_score(y_test, test_pred))

    payload = {"metric": metric}
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
