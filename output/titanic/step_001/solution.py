import json
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "/Users/jiaruixu/work_space/agentic-learn/data/titanic.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Remove known leakage-heavy columns.
    out = out.drop(columns=["boat", "body", "home.dest"])

    # Title extraction and grouping from passenger name.
    out["title"] = (
        out["name"].str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Unknown").str.strip()
    )
    out["title"] = out["title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
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
    })

    # Family-aware features.
    out["family_size"] = out["sibsp"].fillna(0) + out["parch"].fillna(0) + 1
    out["is_alone"] = (out["family_size"] == 1).astype(int)

    # Cabin deck grouping from cabin prefix letter.
    out["cabin_group"] = out["cabin"].fillna("U").astype(str).str[0].replace({"T": "A"})

    # Ticket-based and fare interaction features.
    ticket_clean = out["ticket"].fillna("UNK").astype(str)
    out["ticket_prefix"] = (
        ticket_clean.str.upper().str.replace(r"[^A-Z./ ]", "", regex=True).str.strip().replace("", "NONE")
    )
    ticket_counts = ticket_clean.value_counts()
    out["ticket_group_size"] = ticket_clean.map(ticket_counts).fillna(1)

    out["fare_per_person"] = out["fare"] / out["family_size"].replace(0, 1)
    out["fare_per_person"] = out["fare_per_person"].replace([np.inf, -np.inf], np.nan)

    # Additional interactions/missingness indicators.
    out["name_len"] = out["name"].fillna("").str.len()
    out["class_sex"] = out["pclass"].astype(str) + "_" + out["sex"].astype(str)
    out["age_missing"] = out["age"].isna().astype(int)

    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    X = df.drop(columns=["survived"])
    y = df["survived"]

    numeric_features = [
        "age",
        "sibsp",
        "parch",
        "fare",
        "family_size",
        "fare_per_person",
        "ticket_group_size",
        "name_len",
        "age_missing",
    ]
    categorical_features = [
        "pclass",
        "sex",
        "embarked",
        "title",
        "cabin_group",
        "ticket_prefix",
        "class_sex",
        "is_alone",
    ]

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
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        min_samples_split=4,
        max_features="sqrt",
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1).mean()

    result = {"metric": round(float(score), 6)}

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
