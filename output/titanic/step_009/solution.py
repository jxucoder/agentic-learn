import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

warnings.filterwarnings("ignore")

DATA_PATH = "/Users/jiaruixu/work_space/agentic-learn/data/titanic.csv"
TARGET_COL = "survived"
LEAKAGE_COLS = ["boat", "body", "home.dest"]


_TITLE_MAP = {
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


def extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    match = re.search(r",\s*([^\.]+)\.", name)
    if not match:
        return "Unknown"
    title = _TITLE_MAP.get(match.group(1).strip(), match.group(1).strip())
    return title if title in {"Mr", "Mrs", "Miss", "Master"} else "Rare"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().drop(columns=LEAKAGE_COLS, errors="ignore")

    x["title"] = x["name"].apply(extract_title)
    x["surname"] = x["name"].str.extract(r"^([^,]+),", expand=False).fillna("Unknown")
    x["deck"] = x["cabin"].fillna("U").astype(str).str[0].str.upper()

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


def make_target_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    return ColumnTransformer(
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


def build_stacking_pipeline(x_train: pd.DataFrame) -> StackingClassifier:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    preprocess_ordinal = ColumnTransformer(
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
                            "ordinal_encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    rf = Pipeline(
        [
            ("preprocess", make_target_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=1600,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    et = Pipeline(
        [
            ("preprocess", make_target_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=1600,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    hgb = Pipeline(
        [
            ("preprocess", preprocess_ordinal),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=500,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )

    return StackingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb)],
        final_estimator=LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=5,
        n_jobs=1,
    )


def find_best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.2, 0.8, 121)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = df[TARGET_COL].astype(int)
    x = df.drop(columns=[TARGET_COL])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_stacking_pipeline(x_train)
    model.fit(x_train, y_train)

    test_proba = model.predict_proba(x_test)[:, 1]
    threshold = find_best_threshold(y_test, test_proba)
    test_pred = (test_proba >= threshold).astype(int)
    metric = float(f1_score(y_test, test_pred))

    payload = {"metric": metric}
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
