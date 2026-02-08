"""Code templates for text/NLP competitions."""

TFIDF_LOGREG = '''"""TF-IDF + Logistic Regression for text classification."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Loading data...")
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')
print(f"Train: {{train_df.shape}}, Test: {{test_df.shape}}")

ID_COL = '{id_column}'
TARGET_COL = '{target_column}'
TEXT_COL = '{text_column}'

test_ids = test_df[ID_COL].copy()

# === TF-IDF Vectorization ===
print("TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
X_train = tfidf.fit_transform(train_df[TEXT_COL].fillna(''))
X_test = tfidf.transform(test_df[TEXT_COL].fillna(''))
y = train_df[TARGET_COL]

print(f"TF-IDF features: {{X_train.shape[1]}}")

# === Train ===
print("Training Logistic Regression...")
model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train, y)

cv_scores = cross_val_score(model, X_train, y, cv=5, scoring='{scoring}')
print(f"CV Score: {{cv_scores.mean():.6f}} (+/- {{cv_scores.std()*2:.6f}})")

# === Predict & Submit ===
{predict_code}

import os
os.makedirs('{submission_dir}', exist_ok=True)
submission = pd.DataFrame({{'{id_column}': test_ids, '{target_column}': predictions}})
submission.to_csv(os.path.join('{submission_dir}', 'submission.csv'), index=False)
print(f"Submission saved: {{submission.shape}}")
'''


TFIDF_LGBM = '''"""TF-IDF + LightGBM for text classification."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Loading data...")
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')
print(f"Train: {{train_df.shape}}, Test: {{test_df.shape}}")

ID_COL = '{id_column}'
TARGET_COL = '{target_column}'
TEXT_COL = '{text_column}'

test_ids = test_df[ID_COL].copy()

print("TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
X_train = tfidf.fit_transform(train_df[TEXT_COL].fillna(''))
X_test = tfidf.transform(test_df[TEXT_COL].fillna(''))
y = train_df[TARGET_COL]

print("Training LightGBM...")
try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.1, num_leaves=31,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_STATE)

model.fit(X_train, y)

cv_scores = cross_val_score(model, X_train, y, cv=5, scoring='{scoring}')
print(f"CV Score: {{cv_scores.mean():.6f}} (+/- {{cv_scores.std()*2:.6f}})")

{predict_code}

import os
os.makedirs('{submission_dir}', exist_ok=True)
submission = pd.DataFrame({{'{id_column}': test_ids, '{target_column}': predictions}})
submission.to_csv(os.path.join('{submission_dir}', 'submission.csv'), index=False)
print(f"Submission saved: {{submission.shape}}")
'''
