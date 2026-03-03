# Exploration

## 1) Key patterns and distributions

Dataset:
- Shape: `2000 x 10`
- Target balance: `mean(target)=0.500` (balanced)

Missingness:
- `distance_km`: `5.90%`
- `income`: `5.20%`
- `age`: `5.15%`
- `satisfaction`: `5.15%`
- `hours_worked`: `4.85%`
- Categorical + noise columns have no missing values.

Category distributions:
- `region`: near-uniform (`south 412`, `west 409`, `north 408`, `east 389`, `central 382`)
- `education`: `bachelors 702`, `high_school 686`, `masters 404`, `phd 208`

Numeric behavior and signal:
- `income` is right-skewed (`max=421.19`, median `32.80`) and strongly predictive.
- `distance_km` is right-skewed (`max=94.98`) with a clear monotonic negative relation to target.
- `age` shows non-linearity (better outcomes in middle ages, drop in oldest bin).

Univariate association with target:
- Pearson correlation: `income +0.381`, `distance_km -0.119`, `hours_worked +0.078`, `satisfaction +0.066`, `age +0.029`.
- Mutual information ranking: `education 0.1747`, `income 0.1228`, `distance_km 0.0453`, `region 0.0320`.

Non-linear effects from binned target rates:
- `income` quintiles: `0.141 -> 0.366 -> 0.521 -> 0.664 -> 0.810` (strong monotonic increase)
- `distance_km` bins `(0,2]` to `(40,100]`: `0.720 -> 0.400` (strong decay)
- `age` bins: peak around `45-55` (`0.552`), then drops at `65+` (`0.300`)
- `satisfaction > 3`: `0.532` vs `<=3`: `0.472`

Interaction signal:
- `income * education_ordinal` quintiles: `0.034 -> 0.195 -> 0.496 -> 0.794 -> 0.982`.
- Strong segment effect by `region`: `central 0.707` vs `west 0.337`.
- `region x education` is highly separative (`west+high_school 0.047` vs `central+phd 1.000`).

Noise features:
- `noise_feature_a` and `noise_feature_b` have near-flat target rates across quintiles.
- MI is near zero (`noise_feature_a 0.0000`, `noise_feature_b 0.0084`), so they were treated as noise and removed.

## 2) Feature engineering decisions and why

I used engineered features to capture observed non-linearities and interactions:
- Missing indicators for numeric/ordinal columns (`income`, `age`, `hours_worked`, `distance_km`, `satisfaction`) because missing groups had shifted target rates.
- Non-linear transforms:
  - `log_income`, `income_sq`
  - `age_centered`, `age_centered_sq`
  - `log_hours`, `hours_sq`
  - `log_distance`, `distance_inv`, `distance_sq`
  - `satisfaction_gt3`, `satisfaction_centered`
- Interactions:
  - `hours_div_distance`, `hours_x_distance`
  - `age_x_satisfaction`
  - `income_x_education` with `education` ordinal map (`high_school=1`, `bachelors=2`, `masters=3`, `phd=4`)
  - categorical crosses `region_education` and `region_hours_bucket`
- Noise handling:
  - dropped columns with `"noise"` in name (`noise_feature_a`, `noise_feature_b`).

## 3) Model choice rationale

Final model:
- Preprocessing: median imputation + scaling for numeric features; most-frequent imputation + one-hot encoding for categorical features.
- Estimator: `LogisticRegression(penalty='l2', C=2.0, solver='lbfgs')`.
- Evaluation protocol: 5-fold stratified OOF probabilities with threshold search (`0.2..0.8`) to optimize F1 directly.

Why this model:
- With the engineered features, the class boundary is close to linear in transformed space.
- L2 logistic handled the larger engineered feature set more effectively than sparse L1 in this run.
- Threshold tuning gave a measurable gain over default 0.5 cutoff.

Final metric from `solution.py`:
- `{"metric": 0.9204771371769385}`

## 4) What I tried that didn't work and why

5-fold CV F1 results from alternatives:
- `HistGradientBoosting` variants: `0.8925`, `0.8937`
- `RandomForest` variants: `0.8952`, `0.8941`
- `ExtraTrees` variants: `0.8991`, `0.8971`
- `SVC (RBF)` with threshold tuning: `0.8984`
- `GaussianNB`: `0.7785`

Why they underperformed:
- Tree ensembles underused the continuous/ratio interaction structure compared with engineered linear separation.
- RBF-SVC improved nonlinear fit but did not beat tuned logistic on this sample size.
- Naive Bayes assumptions were too restrictive for the mixed-feature interaction-heavy setting.

For context, stronger linear baselines were:
- `L1 Logistic (C=0.4)` + threshold tuning: `0.9200`
- `L2 Logistic (C=2.0)` + threshold tuning: `0.9205` (selected)
