"""Code templates for tabular ML competitions.

These templates produce complete, runnable Python scripts for common
tabular ML patterns. Used for initial baseline generation and as
starting points for LLM-guided improvements.
"""

XGBOOST_BASELINE = '''"""XGBoost baseline for tabular competition."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === Load Data ===
print("Loading data...")
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')
print(f"Train shape: {{train_df.shape}}")
print(f"Test shape: {{test_df.shape}}")

# === Identify Columns ===
ID_COL = '{id_column}'
TARGET_COL = '{target_column}'

test_ids = test_df[ID_COL].copy()

X = train_df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
y = train_df[TARGET_COL]
X_test = test_df.drop(columns=[ID_COL, TARGET_COL], errors='ignore')

# === Preprocessing ===
print("Preprocessing...")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Handle missing numeric
for col in num_cols:
    median = X[col].median()
    X[col] = X[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

# Encode categoricals
label_encoders = {{}}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# === Train XGBoost ===
print("Training XGBoost...")
try:
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss',
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    print("XGBoost not available, using GradientBoosting")
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    )

model.fit(X, y)

# === Cross Validation ===
print("Cross-validating...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='{scoring}')
print(f"CV Score: {{cv_scores.mean():.6f}} (+/- {{cv_scores.std()*2:.6f}})")

# === Predict & Submit ===
print("Making predictions...")
{predict_code}

import os
os.makedirs('{submission_dir}', exist_ok=True)
submission = pd.DataFrame({{'{id_column}': test_ids, '{target_column}': predictions}})
submission.to_csv(os.path.join('{submission_dir}', 'submission.csv'), index=False)
print(f"Submission saved: {{submission.shape}}")
print(submission.head())
'''


LIGHTGBM_BASELINE = '''"""LightGBM baseline for tabular competition."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === Load Data ===
print("Loading data...")
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')
print(f"Train shape: {{train_df.shape}}")
print(f"Test shape: {{test_df.shape}}")

# === Identify Columns ===
ID_COL = '{id_column}'
TARGET_COL = '{target_column}'

test_ids = test_df[ID_COL].copy()
X = train_df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
y = train_df[TARGET_COL]
X_test = test_df.drop(columns=[ID_COL, TARGET_COL], errors='ignore')

# === Preprocessing ===
print("Preprocessing...")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

for col in num_cols:
    median = X[col].median()
    X[col] = X[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

X_test = X_test.reindex(columns=X.columns, fill_value=0)

# === Train LightGBM ===
print("Training LightGBM...")
try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=-1, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("LightGBM not available, using HistGradientBoosting")
    model = HistGradientBoostingClassifier(
        max_iter=500, max_depth=None, learning_rate=0.05, random_state=RANDOM_STATE
    )

model.fit(X, y)

# === Cross Validation ===
print("Cross-validating...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='{scoring}')
print(f"CV Score: {{cv_scores.mean():.6f}} (+/- {{cv_scores.std()*2:.6f}})")

# === Predict & Submit ===
print("Making predictions...")
{predict_code}

import os
os.makedirs('{submission_dir}', exist_ok=True)
submission = pd.DataFrame({{'{id_column}': test_ids, '{target_column}': predictions}})
submission.to_csv(os.path.join('{submission_dir}', 'submission.csv'), index=False)
print(f"Submission saved: {{submission.shape}}")
print(submission.head())
'''


RANDOM_FOREST_BASELINE = '''"""Random Forest baseline for tabular competition."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Loading data...")
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')
print(f"Train: {{train_df.shape}}, Test: {{test_df.shape}}")

ID_COL = '{id_column}'
TARGET_COL = '{target_column}'
test_ids = test_df[ID_COL].copy()

X = train_df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
y = train_df[TARGET_COL]
X_test = test_df.drop(columns=[ID_COL, TARGET_COL], errors='ignore')

# Preprocessing
for col in X.select_dtypes(include=[np.number]).columns:
    median = X[col].median()
    X[col] = X[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

for col in X.select_dtypes(include=['object', 'category', 'bool']).columns:
    le = LabelEncoder()
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X, y)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='{scoring}')
print(f"CV Score: {{cv_scores.mean():.6f}} (+/- {{cv_scores.std()*2:.6f}})")

{predict_code}

import os
os.makedirs('{submission_dir}', exist_ok=True)
submission = pd.DataFrame({{'{id_column}': test_ids, '{target_column}': predictions}})
submission.to_csv(os.path.join('{submission_dir}', 'submission.csv'), index=False)
print(f"Submission saved: {{submission.shape}}")
'''


def get_predict_code(prediction_type: str = "labels") -> str:
    """Get prediction code based on prediction type."""
    if prediction_type in ("probabilities", "proba"):
        return """if hasattr(model, 'predict_proba'):
    predictions = model.predict_proba(X_test)[:, 1]
else:
    predictions = model.predict(X_test)"""
    else:
        return "predictions = model.predict(X_test)"


def get_scoring(metric_name: str) -> str:
    """Map metric name to sklearn scoring string."""
    mapping = {
        "Accuracy": "accuracy",
        "AUC-ROC": "roc_auc",
        "F1 Score": "f1",
        "Macro F1": "f1_macro",
        "Log Loss": "neg_log_loss",
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2",
    }
    return mapping.get(metric_name, "accuracy")
