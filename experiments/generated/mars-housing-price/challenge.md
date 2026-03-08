# Martian Market Dynamics: Predicting Mars Housing Prices

Predict the elusive target value representing housing prices on Mars, navigating complex tabular data with distribution shifts and intricate feature interactions to achieve robust R2 scores.

## Scenario
Welcome to the Martian Frontier! As humanity expands its reach to the red planet, the burgeoning real estate market on Mars presents both immense opportunity and formidable challenges. Your mission, should you choose to accept it, is to aid the Interplanetary Real Estate Council (IREC) in establishing a stable and predictable housing market. Accurate price prediction is crucial for settlers, investors, and the sustainable growth of Martian colonies. Competitors will analyze a rich dataset detailing various factors influencing property values across nascent Martian settlements.

## Objective
The primary objective is to develop a robust regression model capable of accurately predicting the `target` column, which represents the dynamic housing price index on Mars. Submissions will be evaluated based on the R2 coefficient of determination.

## Data Highlights
- Diverse feature set: Includes `event_date`, `customer_id`, `income`, `age`, `tenure_months`, `sessions_30d`, `avg_order_value`, `discount_rate`, `support_wait_minutes`, `return_rate`, `complaints_90d`, `marketing_touches_30d`, `region`, `segment`, `plan_tier`, `acquisition_channel`, `city_code`, `campaign_id`, `campaign_quality_proxy`, `aux_metric_01`, and `aux_bucket_02`.
- Temporal insights: The `event_date` column hints at potential time-series dependencies and evolving market conditions.
- High-cardinality categories: Features like `customer_id`, `city_code`, and `campaign_id` introduce significant challenges due to their unique or numerous distinct values.
- Missing data patterns: Expect various patterns of missing values across features, requiring careful imputation or handling strategies.
- Auxiliary metrics: `aux_metric_01` and `aux_bucket_02` provide additional, potentially cryptic, information that could be crucial.

## Modeling Challenges
- Distribution Shift: The training and test datasets exhibit subtle yet significant distribution shifts, demanding models robust to evolving Martian market dynamics.
- Feature Engineering: Unlocking predictive power will require advanced feature engineering from raw inputs, especially for categorical and temporal features.
- High-Cardinality Handling: Effectively encoding and utilizing high-cardinality categorical features without overfitting is a key challenge.
- Robust Validation Strategy: A robust local validation scheme is paramount to ensure model performance generalizes to the unseen public and private leaderboards.
- Missingness Patterns: Identifying and addressing non-random missing data patterns is critical for preventing bias and improving predictive accuracy.
- Ensemble Potential: Combining diverse model architectures and predictions may yield superior performance due to the multifaceted nature of the data.

## Submission Requirements
- Submissions must be a CSV file with exactly two columns: `row_id` and `target`.
- The `row_id` column should correspond to the `row_id` from the test dataset.
- The `target` column should contain your predicted housing price values for each `row_id`.
- Ensure your submission file adheres strictly to the format of the provided `synth_mars_housing_price_sample_submission.csv` to avoid submission errors.
- Predictions should be numerical (float or integer).

## Submission Format
- Final file name: `submission.csv`
- Required columns: `row_id`, `target`
- Include every row from `synth_mars_housing_price_test.csv` exactly once
- Match the column order shown in `synth_mars_housing_price_sample_submission.csv`
- Row order is optional, but duplicate or missing `row_id` values are invalid


## Public Validation
- Fixed labeled validation split: `synth_mars_housing_price_validation.csv`
- Validation prediction template: `synth_mars_housing_price_validation_sample_submission.csv`
- Use the validation split for model selection and local score tracking
To score validation predictions, run:

```bash
uv run python evaluate_validation.py --submission validation_submission.csv
```



## Local Validation
From the experiment root, run:

```bash
uv run python validate_submission.py --submission submission.csv
```

The validator checks schema, row coverage, duplicate `row_id` values, and basic target parsing before leaderboard evaluation.


## Evaluation
Submissions will be evaluated using the R2 coefficient of determination. A higher R2 score indicates a better fit of the model to the observed data, with 1.0 being a perfect prediction. The final leaderboard will be determined by the R2 score on a hidden test set.
