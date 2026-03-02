import json
import re
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path('/Users/jiaruixu/work_space/agentic-learn/data/titanic.csv')
RESULT_PATH = Path('result.json')
RANDOM_STATE = 42


def extract_title(name: str) -> str:
    if pd.isna(name):
        return 'Unknown'
    match = re.search(r',\s*([^\.]+)\.', str(name))
    if not match:
        return 'Unknown'

    title = match.group(1).strip()
    title = title.replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')

    if title in {'Lady', 'Countess', 'Dona', 'Jonkheer', 'Sir', 'Don'}:
        return 'Nobility'
    if title in {'Capt', 'Col', 'Major'}:
        return 'Officer'
    if title in {'Dr', 'Rev'}:
        return 'Professional'
    if title not in {'Mr', 'Miss', 'Mrs', 'Master'}:
        return 'Rare'
    return title


def cabin_group(cabin: str) -> str:
    if pd.isna(cabin):
        return 'U'
    match = re.search(r'([A-Za-z])', str(cabin))
    if not match:
        return 'U'

    letter = match.group(1).upper()
    if letter in {'A', 'B', 'C'}:
        return 'ABC'
    if letter in {'D', 'E'}:
        return 'DE'
    if letter in {'F', 'G', 'T'}:
        return 'FGT'
    return 'U'


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=['survived']).copy()

    # Leak-prone columns requested to be removed.
    X = X.drop(columns=['boat', 'body', 'home.dest'], errors='ignore')

    X['title'] = X['name'].map(extract_title)
    X['family_size'] = X['sibsp'].fillna(0) + X['parch'].fillna(0) + 1
    X['is_alone'] = (X['family_size'] == 1).astype(int)
    X['cabin_group'] = X['cabin'].map(cabin_group)
    X['fare_per_person'] = X['fare'] / X['family_size'].replace(0, 1)

    X = X.drop(columns=['name', 'cabin'])
    return X


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    y = df['survived'].astype(int)
    X = build_features(df)

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]

    preprocessor_lr = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), num_cols),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    models = {
        'logreg': Pipeline(
            steps=[
                ('preprocessor', preprocessor_lr),
                (
                    'model',
                    LogisticRegression(
                        max_iter=3000,
                        class_weight='balanced',
                        solver='liblinear',
                        C=1.2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        'random_forest': Pipeline(
            steps=[
                ('preprocessor', preprocessor_tree),
                (
                    'model',
                    RandomForestClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        class_weight='balanced_subsample',
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        'extra_trees': Pipeline(
            steps=[
                ('preprocessor', preprocessor_tree),
                (
                    'model',
                    ExtraTreesClassifier(
                        n_estimators=900,
                        min_samples_leaf=2,
                        class_weight='balanced',
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_metric = float('-inf')
    for model in models.values():
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=1)
        mean_score = float(scores.mean())
        if mean_score > best_metric:
            best_metric = mean_score

    result = {'metric': round(best_metric, 6)}
    RESULT_PATH.write_text(json.dumps(result), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
