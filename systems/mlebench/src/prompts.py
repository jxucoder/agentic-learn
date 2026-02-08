"""System prompts and prompt templates for MLE-bench competition solving."""

from __future__ import annotations

from src.task_parser import TaskInfo


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert machine learning engineer competing in an offline Kaggle competition via MLE-bench.

## Your Goal
Write a complete Python script that:
1. Loads the training data and test data
2. Trains a model
3. Generates predictions on the test set
4. Saves a valid submission file to the specified path

## Critical Rules
- You MUST produce a valid submission CSV. A failed submission scores zero.
- Always write a COMPLETE, SELF-CONTAINED Python script. No fragments.
- The script must be runnable end-to-end with `python solution.py`.
- Print your cross-validation score so we can track progress.
- Handle errors gracefully -- if a complex approach fails, fall back to simpler methods.
- Do NOT use test labels or leak data. Train only on training data.
- The submission file must match the required format exactly.

## Output Format
Return ONLY the Python code inside a single ```python code block.
Do not include any explanation before or after the code block.
The code must be complete and self-contained.
"""


def build_initial_prompt(task: TaskInfo) -> str:
    """Build the initial prompt for generating a baseline solution."""
    return f"""## Competition Details

{task.description[:8000]}

## Data Summary

{task.to_context()}

## Submission Requirements

- Save your submission to: `{task.submission_dir}/submission.csv`
- The submission must be a CSV file
- Required columns: {', '.join(task.submission_columns) if task.submission_columns else 'See competition description'}
- Metric: {task.metric_name} ({'higher is better' if task.higher_is_better else 'lower is better'})

## Data Paths

- Data directory: `{task.data_dir}`
- Train file: `{task.train_file.path if task.train_file else 'See data directory'}`
- Test file: `{task.test_file.path if task.test_file else 'See data directory'}`
- Sample submission: `{task.sample_submission_file.path if task.sample_submission_file else 'N/A'}`

## Instructions

Write a complete Python script that:
1. Reads the training and test data
2. Performs basic preprocessing
3. Trains a solid baseline model (e.g., XGBoost, LightGBM, or Random Forest for tabular data)
4. Generates predictions on the test set
5. Saves predictions to `{task.submission_dir}/submission.csv` in the required format
6. Prints the cross-validation score

Start with a reliable baseline that definitely produces a valid submission, then we can improve it.
"""


def build_improvement_prompt(
    task: TaskInfo,
    previous_code: str,
    previous_score: str,
    previous_stdout: str,
    iteration: int,
    best_score: str,
) -> str:
    """Build a prompt for improving a previous solution."""
    return f"""## Competition: {task.competition_id}
Metric: {task.metric_name} ({'higher is better' if task.higher_is_better else 'lower is better'})

## Previous Solution (Iteration {iteration})
CV Score: {previous_score}
Best Score So Far: {best_score}

### Previous Output:
```
{previous_stdout[-3000:]}
```

### Previous Code:
```python
{previous_code[-6000:]}
```

## Instructions

Improve the solution. Consider:
- Better feature engineering
- More sophisticated model (ensemble, stacking)
- Hyperparameter tuning
- Better preprocessing
- Different model architecture
- Data augmentation if applicable

Write a COMPLETE, self-contained Python script that saves to `{task.submission_dir}/submission.csv`.
The script must be runnable end-to-end.
Print your CV score clearly.
"""


def build_error_fix_prompt(
    task: TaskInfo,
    previous_code: str,
    error_output: str,
    iteration: int,
) -> str:
    """Build a prompt for fixing an error in a previous solution."""
    return f"""## Competition: {task.competition_id}

## Error in Previous Solution (Iteration {iteration})

The following code produced an error:

### Code:
```python
{previous_code[-6000:]}
```

### Error Output:
```
{error_output[-3000:]}
```

## Instructions

Fix the error and produce a working solution.
- Carefully read the error message
- Fix the specific issue
- If the approach is fundamentally broken, try a simpler approach
- Make sure the output is a COMPLETE, self-contained Python script
- Save submission to `{task.submission_dir}/submission.csv`
- Print the CV score
"""


def build_debug_prompt(
    task: TaskInfo,
    previous_code: str,
    stdout: str,
    stderr: str,
) -> str:
    """Build a prompt for debugging a solution that ran but may have issues."""
    return f"""## Competition: {task.competition_id}

The solution ran but may have issues. Here is the output:

### stdout:
```
{stdout[-2000:]}
```

### stderr:
```
{stderr[-2000:]}
```

### Code:
```python
{previous_code[-6000:]}
```

## Instructions

Analyze the output and fix any issues:
- Check if the submission file was created correctly
- Check if the submission format matches requirements
- Fix any warnings that might indicate data issues
- Ensure predictions are in the correct format

Write a COMPLETE, self-contained Python script.
Save submission to `{task.submission_dir}/submission.csv`.
"""
