"""Multi-model support and solution ensembling for MLE-bench.

Phase 4 features:
1. Multi-model LLM configuration (different models for different roles)
2. Solution ensembling (combine predictions from multiple MCTS branches)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
import logging

from src.llm import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# Multi-Model Configuration
# =============================================================================

@dataclass
class ModelRole:
    """Configuration for a specific model role."""
    provider: str = "anthropic"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 16384
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class MultiModelConfig:
    """Configuration for multi-model setup.

    Different models can be used for different roles:
    - reasoner: Strategic planning and complex code generation
    - coder: Routine code generation and bug fixes
    - feedback: Reflection and analysis
    """
    reasoner: ModelRole = field(default_factory=lambda: ModelRole(
        provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.7
    ))
    coder: ModelRole = field(default_factory=lambda: ModelRole(
        provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.5
    ))
    feedback: ModelRole = field(default_factory=lambda: ModelRole(
        provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.3, max_tokens=2048
    ))


class MultiModelClient:
    """Client that routes requests to different LLMs based on role.

    Usage:
        client = MultiModelClient(config)
        code = client.generate_code("write XGBoost solution", role="coder")
        analysis = client.complete("analyze this error", role="feedback")
    """

    def __init__(self, config: Optional[MultiModelConfig] = None):
        self.config = config or MultiModelConfig()
        self._clients: dict[str, LLMClient] = {}

    def _get_client(self, role: str) -> LLMClient:
        """Get or create LLM client for a role."""
        if role not in self._clients:
            role_config = getattr(self.config, role, self.config.coder)
            self._clients[role] = LLMClient(
                provider=role_config.provider,
                model=role_config.model,
                api_key=role_config.api_key,
                base_url=role_config.base_url,
                temperature=role_config.temperature,
                max_tokens=role_config.max_tokens,
            )
        return self._clients[role]

    def generate_code(self, prompt: str, system: Optional[str] = None, role: str = "coder") -> str:
        """Generate code using the specified role's model."""
        client = self._get_client(role)
        return client.generate_code(prompt, system=system)

    def complete(self, prompt: str, system: Optional[str] = None, role: str = "feedback") -> str:
        """Complete a prompt using the specified role's model."""
        client = self._get_client(role)
        response = client.complete(prompt, system=system)
        return response.content


# =============================================================================
# Solution Ensembling
# =============================================================================

ENSEMBLE_PROMPT = """You are an expert ML engineer. Given multiple solution scripts that each
produce predictions for a Kaggle competition, write a SINGLE ensemble script that:

1. Reads the individual submission files produced by each solution
2. Combines them using averaging (for regression/probabilities) or voting (for classification)
3. Saves the final ensembled submission

The individual solution submission files are at:
{submission_paths}

Competition metric: {metric_name} ({'higher is better' if {higher_is_better} else 'lower is better'})
Submission format: columns {submission_columns}
Save ensemble submission to: {submission_dir}/submission.csv

Write a COMPLETE Python script. Return only the code in a ```python block.
"""


def generate_ensemble_code(
    submission_paths: list[str],
    scores: list[float],
    metric_name: str,
    higher_is_better: bool,
    submission_columns: list[str],
    submission_dir: str,
    prediction_type: str = "labels",
) -> str:
    """Generate code to ensemble multiple submissions.

    Uses score-weighted averaging for a simple but effective ensemble.

    Args:
        submission_paths: Paths to individual submission CSVs.
        scores: Scores for each submission (for weighting).
        metric_name: Competition metric name.
        higher_is_better: Whether higher metric is better.
        submission_columns: Expected columns in submission.
        submission_dir: Directory for final submission.
        prediction_type: "labels", "probabilities", or "values".

    Returns:
        Complete Python script for ensembling.
    """
    id_col = submission_columns[0] if submission_columns else "id"
    target_cols = submission_columns[1:] if len(submission_columns) > 1 else ["target"]

    # Normalize scores to weights
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            if higher_is_better:
                weights = [(s - min_score) / (max_score - min_score) + 0.1 for s in scores]
            else:
                weights = [(max_score - s) / (max_score - min_score) + 0.1 for s in scores]
        else:
            weights = [1.0] * len(scores)
        total = sum(weights)
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(submission_paths)] * len(submission_paths)

    paths_str = "\n".join(f'    "{p}",  # weight={w:.3f}, score={s}' for p, w, s in zip(submission_paths, weights, scores))
    weights_str = ", ".join(f"{w:.4f}" for w in weights)
    target_cols_str = ", ".join(f'"{c}"' for c in target_cols)

    code = f'''"""Ensemble of {len(submission_paths)} solutions."""
import pandas as pd
import numpy as np
import os

os.makedirs("{submission_dir}", exist_ok=True)

# Individual submissions and their weights (based on CV scores)
submission_paths = [
{paths_str}
]
weights = [{weights_str}]

print(f"Ensembling {{len(submission_paths)}} submissions...")

# Load all submissions
dfs = []
for path in submission_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded {{path}}: {{df.shape}}")
    else:
        print(f"  WARNING: {{path}} not found, skipping")

if not dfs:
    print("ERROR: No submissions found to ensemble")
    exit(1)

# Use first submission as base
ensemble = dfs[0][["{id_col}"]].copy()

# Weighted average for each target column
target_cols = [{target_cols_str}]
for col in target_cols:
    weighted_sum = np.zeros(len(ensemble))
    total_weight = 0
    for df, w in zip(dfs, weights[:len(dfs)]):
        if col in df.columns:
            weighted_sum += df[col].values * w
            total_weight += w
    if total_weight > 0:
        ensemble[col] = weighted_sum / total_weight
    else:
        ensemble[col] = dfs[0][col]

'''

    if prediction_type == "labels":
        code += f'''
# Round to nearest class for classification
for col in target_cols:
    # Check if values are binary/integer
    unique_vals = ensemble[col].unique()
    if len(unique_vals) <= 20:  # Likely classification
        ensemble[col] = ensemble[col].round().astype(int)
'''

    code += f'''
# Save ensemble
output_path = os.path.join("{submission_dir}", "submission.csv")
ensemble.to_csv(output_path, index=False)
print(f"Ensemble submission saved: {{ensemble.shape}}")
print(ensemble.head())
'''

    return code


def generate_stacking_code(
    submission_paths: list[str],
    train_pred_paths: list[str],
    train_labels_path: str,
    metric_name: str,
    submission_columns: list[str],
    submission_dir: str,
) -> str:
    """Generate code for stacking ensemble (meta-learner).

    More sophisticated than simple averaging -- trains a meta-model
    on out-of-fold predictions from base models.

    Args:
        submission_paths: Paths to test predictions from each model.
        train_pred_paths: Paths to out-of-fold train predictions.
        train_labels_path: Path to training labels.
        metric_name: Competition metric.
        submission_columns: Expected submission columns.
        submission_dir: Output directory.

    Returns:
        Complete Python script for stacking.
    """
    id_col = submission_columns[0] if submission_columns else "id"
    target_col = submission_columns[1] if len(submission_columns) > 1 else "target"

    code = f'''"""Stacking ensemble with meta-learner."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import os

os.makedirs("{submission_dir}", exist_ok=True)

# Load base model predictions (out-of-fold for train, full for test)
train_preds = []
test_preds = []

train_pred_paths = {train_pred_paths}
test_pred_paths = {submission_paths}

for tp, sp in zip(train_pred_paths, test_pred_paths):
    if os.path.exists(tp) and os.path.exists(sp):
        train_preds.append(pd.read_csv(tp)["{target_col}"].values)
        test_preds.append(pd.read_csv(sp)["{target_col}"].values)

if not train_preds:
    print("ERROR: No predictions found for stacking")
    exit(1)

X_meta_train = np.column_stack(train_preds)
X_meta_test = np.column_stack(test_preds)

# Load true labels
y_train = pd.read_csv("{train_labels_path}")["{target_col}"].values

# Train meta-learner
meta_model = LogisticRegression(C=1.0, max_iter=1000)
cv_scores = cross_val_score(meta_model, X_meta_train, y_train, cv=5, scoring="accuracy")
print(f"Stacking CV Score: {{cv_scores.mean():.6f}}")

meta_model.fit(X_meta_train, y_train)
meta_predictions = meta_model.predict(X_meta_test)

# Save
test_df = pd.read_csv(test_pred_paths[0])
submission = pd.DataFrame({{"{id_col}": test_df["{id_col}"], "{target_col}": meta_predictions}})
submission.to_csv(os.path.join("{submission_dir}", "submission.csv"), index=False)
print(f"Stacking submission: {{submission.shape}}")
'''

    return code
