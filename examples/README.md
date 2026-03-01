# Benchmarks

Two scikit-learn tasks that exercise the agent on the problems
that matter: mixed types, missing values, feature engineering,
and model selection.

## Titanic — binary classification

```bash
python examples/titanic.py
```

| What makes it hard | Score range |
|---|---|
| Mixed types (numeric + categorical) | Naive: ~0.72 F1 |
| ~20% missing ages, ~77% missing cabin | Baseline (logistic regression): ~0.77 F1 |
| Feature engineering: title, family size, deck | Good: ~0.82+ F1 |

The agent should discover that extracting titles from names,
engineering family-size features, and using tree-based models
outperforms a vanilla classifier.

## California Housing — regression

```bash
python examples/california_housing.py
```

| What makes it hard | Score range |
|---|---|
| Spatial structure (lat/lon encode geography) | Naive: ~0.60 R² |
| Feature interactions (income × location) | Baseline (ridge regression): ~0.60 R² |
| Ratio features: bedrooms/rooms, pop/households | Good: ~0.85+ R² |

The agent should discover that gradient boosting with location-aware
features dramatically outperforms linear models on this dataset.

## Running

Each benchmark downloads its dataset on first run (cached in `data/`),
then runs 10 agent steps. Results go to `output/<benchmark>/`.

```
output/titanic/
    journal.jsonl         full experiment history
    best_solution.py      highest-scoring script
    step_000/             agent working directory for step 0
    step_001/             ...
```

Set `MLE_MODEL` to use a different model:

```bash
MLE_MODEL=o3 python examples/titanic.py
```
