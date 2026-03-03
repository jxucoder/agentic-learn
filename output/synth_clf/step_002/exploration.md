# Exploration

## 1) Key patterns, distributions, and missingness

Dataset shape: `2000 x 10` (9 features + target), target is balanced (`target mean = 0.500`).

Missingness (fraction missing):
- `distance_km`: `0.059`
- `income`: `0.052`
- `age`: `0.0515`
- `satisfaction`: `0.0515`
- `hours_worked`: `0.0485`
- `region`, `education`, `noise_feature_a`, `noise_feature_b`: `0.0`

Category distributions:
- `region`: near-uniform across 5 levels (`south 412`, `west 409`, `north 408`, `east 389`, `central 382`)
- `education`: `bachelors 702`, `high_school 686`, `masters 404`, `phd 208`

Strong target signal (univariate):
- Mutual information ranking: `education (0.1783)`, `income (0.1178)`, `distance_km (0.0441)`, `region (0.0414)`
- Noise candidates by MI: `noise_feature_a (0.0000)`, `noise_feature_b (0.0034)`

Observed non-linear / interaction structure from binned rates:
- `age` is curved (peak around middle age):
  - `(45,55]` bin has highest positive rate (`0.552`), older tail `(65,80]` drops (`0.300`)
- `satisfaction` behaves threshold-like:
  - `satisfaction > 3`: positive rate `0.532`
  - `satisfaction <= 3`: positive rate `0.472`
- Strong `income x education` interaction:
  - Quintiles of `income * education_ordinal` increase monotonically from `0.034` to `0.982`
- `log(hours_worked)/log(distance_km)` interaction is informative:
  - Top quintile positive rate `0.673` vs lower quintiles around `0.40-0.49`
- Region baseline shifts are large:
  - `central 0.707`, `north 0.556`, `east 0.496`, `south 0.417`, `west 0.337`

Noise behavior check (quintile target rates nearly flat):
- `noise_feature_a`: around `0.483-0.530`
- `noise_feature_b`: around `0.482-0.519`

## 2) Feature engineering decisions and why

I engineered features to directly match observed signal:
- Dropped columns with `"noise"` in name (`noise_feature_a`, `noise_feature_b`) due near-zero signal and flat target rates.
- Added missing indicators for continuous/ordinal features (`income`, `age`, `hours_worked`, `distance_km`, `satisfaction`) because missing subsets had slightly different target rates.
- Non-linear transforms:
  - `age_centered`, `age_centered_sq` for age curvature around ~45.
  - `log_hours`, `log_distance`, `distance_inv` for skew and distance decay.
  - `satisfaction_gt3`, `satisfaction_centered` for threshold and ordinal effect.
- Interactions:
  - `income_x_education` using ordinal map `{high_school:1, bachelors:2, masters:3, phd:4}`.
  - `log_hours_div_log_distance` to capture productivity/commute interaction.
  - `region_education` categorical cross to capture segment baseline shifts.

## 3) Model choice rationale

Final model: **L1-regularized Logistic Regression** (`liblinear`, `C=0.2`) on:
- median-imputed + standardized numeric features
- most-frequent-imputed + one-hot encoded categorical features

Why this model:
- Engineered features make the decision boundary close to linear in transformed space.
- L1 regularization helps prune redundant engineered terms and keeps robustness on a small dataset.
- Sparse one-hot + linear model performed better than tree ensembles in CV here.
- Added probability threshold tuning (OOF predictions, threshold grid `0.2..0.8`) to optimize `f1` directly.

Final score from `solution.py`:
- `{"metric": 0.9140049140049139}`

## 4) What I tried that did not work and why

Tried alternatives (5-fold CV F1):
- ExtraTrees (`500 trees`, deep): `0.8906`
- ExtraTrees (`700 trees`, depth 12): `0.8794`
- HistGradientBoosting variants: `0.8907`, `0.8953`, `0.8965`
- Logistic regression with weaker feature sparsity (`L2`) was slightly worse (`~0.908-0.909`) than L1 (`~0.910` before threshold tuning).

Why they underperformed:
- Tree models did not exploit the hand-crafted ratio/cross effects as efficiently at this sample size.
- L2 logistic kept more weak/redundant terms; L1 gave better bias-variance tradeoff for F1.
- Default threshold (`0.5`) was not optimal for F1; tuned threshold (~`0.41`) improved score.
