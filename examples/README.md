# Benchmarks

## ⚠️ Synthetic benchmarks (recommended)

Public datasets like Titanic and California Housing are present in LLM
training data. When using frontier models (GPT-4, o3, Codex, etc.), any
score improvement could reflect **memorized solutions** rather than
genuine feature engineering. Use synthetic benchmarks for honest evaluation.

```bash
# Binary classification — fresh data each seed
python examples/synth_classification.py
python examples/synth_classification.py --seed 123 --steps 10

# Regression
python examples/synth_regression.py
python examples/synth_regression.py --seed 456 --noise 0.3
```

The synthetic generator (`aglearn.synth`) creates datasets with:
- **Known ground truth** — non-linear interactions the agent must discover
- **Noise features** — two columns that should be ignored
- **Missing values** — configurable fraction of NaNs
- **Unique per seed** — the LLM has never seen the data

See `synth_classification.py` and `synth_regression.py` for full options
(`--seed`, `--steps`, `--samples`, `--noise`).

---

## Legacy benchmarks (public data — may have knowledge leakage)

These benchmarks use well-known public datasets. They are useful for
quick smoke tests and comparison with published results, but **scores
may be inflated** due to data contamination in modern LLMs.

### Titanic — binary classification

```bash
python examples/titanic.py
```

| What makes it hard | Score range |
|---|---|
| Mixed types (numeric + categorical) | Naive: ~0.72 F1 |
| ~20% missing ages, ~77% missing cabin | Baseline (logistic regression): ~0.77 F1 |
| Feature engineering: title, family size, deck | Good: ~0.82+ F1 |

### California Housing — regression

```bash
python examples/california_housing.py
```

| What makes it hard | Score range |
|---|---|
| Spatial structure (lat/lon encode geography) | Naive: ~0.60 R² |
| Feature interactions (income × location) | Baseline (ridge regression): ~0.60 R² |
| Ratio features: bedrooms/rooms, pop/households | Good: ~0.85+ R² |

---

## Running

Each benchmark downloads/generates its dataset on first run (cached in
`data/`), then runs 10 agent steps. Results go to `output/<benchmark>/`.

```
output/<benchmark>/
    journal.jsonl         full experiment history
    best_solution.py      highest-scoring script
    step_000/             agent working directory for step 0
    step_001/             ...
```

Set `MLE_MODEL` to use a different model:

```bash
MLE_MODEL=codex-mini python examples/titanic.py
```
