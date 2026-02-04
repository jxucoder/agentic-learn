#!/usr/bin/env python3
"""Practical DS workflow examples for agentic-learn.

These show real-world data science tasks the agent can help with.
"""

import asyncio
from agentic_learn.core import Agent, AgentConfig, EventType, ToolContext
from agentic_learn.tools import get_tools, PythonTool


# =============================================================================
# Example 1: Data Exploration
# =============================================================================

EXPLORE_DATA_PROMPT = """
I have a dataset. Please help me explore it:

1. Load the data from 'data/titanic.csv'
2. Show the first few rows
3. Check for missing values
4. Show basic statistics
5. Identify the target variable for prediction
"""


# =============================================================================
# Example 2: Feature Engineering
# =============================================================================

FEATURE_ENGINEERING_PROMPT = """
Help me create features for a machine learning model:

1. Read the data from 'data/sales.csv'
2. Create date-based features (day of week, month, quarter)
3. Create lag features for the sales column
4. Handle any missing values appropriately
5. Save the processed data to 'data/sales_features.csv'
"""


# =============================================================================
# Example 3: Model Training
# =============================================================================

TRAIN_MODEL_PROMPT = """
Train a classification model:

1. Load 'data/iris.csv'
2. Split into train/test sets (80/20)
3. Train a Random Forest classifier
4. Evaluate with accuracy, precision, recall
5. Save the model to 'models/iris_rf.pkl'

Use sklearn and show me the code.
"""


# =============================================================================
# Example 4: Hyperparameter Tuning
# =============================================================================

HYPERPARAMETER_PROMPT = """
Help me tune hyperparameters for a gradient boosting model:

1. Load the dataset from 'data/housing.csv'
2. Define a parameter grid for XGBoost:
   - max_depth: [3, 5, 7]
   - learning_rate: [0.01, 0.1, 0.3]
   - n_estimators: [100, 200]
3. Use GridSearchCV with 5-fold cross-validation
4. Report the best parameters and score
"""


# =============================================================================
# Example 5: Visualization
# =============================================================================

VISUALIZATION_PROMPT = """
Create visualizations for my analysis:

1. Load 'data/stocks.csv' with columns: date, price, volume
2. Create a line plot of price over time
3. Create a bar chart of monthly average volume
4. Create a correlation heatmap if there are multiple numeric columns
5. Save all plots to the 'plots/' directory
"""


# =============================================================================
# Example 6: End-to-End ML Pipeline
# =============================================================================

PIPELINE_PROMPT = """
Build a complete ML pipeline for customer churn prediction:

1. Load 'data/churn.csv'
2. Exploratory analysis:
   - Check class balance
   - Identify important features
3. Preprocessing:
   - Handle missing values
   - Encode categorical variables
   - Scale numeric features
4. Model training:
   - Try Logistic Regression and Random Forest
   - Compare using cross-validation
5. Evaluation:
   - Confusion matrix
   - ROC curve
   - Feature importance
6. Save the best model
"""


# =============================================================================
# Runnable Examples (no API needed)
# =============================================================================

async def run_python_code():
    """Example: Run Python code directly (no external deps)."""
    print("=" * 60)
    print("Running Python Code Directly")
    print("=" * 60)

    python_tool = PythonTool()
    ctx = ToolContext(cwd="/tmp", agent=None)

    # Simple data analysis without pandas
    code = """
import random
import statistics

# Create sample data
random.seed(42)
ages = [random.randint(18, 70) for _ in range(100)]
incomes = [random.gauss(50000, 15000) for _ in range(100)]
purchased = [random.choice([0, 1]) for _ in range(100)]

print("Dataset: 100 samples")
print(f"\\nAge Statistics:")
print(f"  Mean: {statistics.mean(ages):.1f}")
print(f"  Std:  {statistics.stdev(ages):.1f}")
print(f"  Min:  {min(ages)}, Max: {max(ages)}")

print(f"\\nIncome Statistics:")
print(f"  Mean: ${statistics.mean(incomes):,.0f}")
print(f"  Std:  ${statistics.stdev(incomes):,.0f}")

print(f"\\nTarget Distribution:")
purchased_pct = sum(purchased) / len(purchased)
print(f"  Purchased: {purchased_pct:.1%}")
print(f"  Not purchased: {1-purchased_pct:.1%}")
"""

    result = await python_tool.execute(ctx, code=code)
    print(result.content)
    print()


async def run_model_training():
    """Example: Simple classifier from scratch."""
    print("=" * 60)
    print("Training a Simple Model (from scratch)")
    print("=" * 60)

    python_tool = PythonTool()
    ctx = ToolContext(cwd="/tmp", agent=None)

    code = """
import random
import math
from collections import Counter

# Simple k-NN classifier from scratch
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        # Calculate distances
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = math.sqrt(sum((a-b)**2 for a,b in zip(x, x_train)))
            distances.append((dist, self.y_train[i]))

        # Get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = [d[1] for d in distances[:self.k]]

        # Vote
        return Counter(k_nearest).most_common(1)[0][0]

# Generate sample data (2 classes, 2 features)
random.seed(42)
X, y = [], []

# Class 0: centered around (2, 2)
for _ in range(50):
    X.append([random.gauss(2, 0.5), random.gauss(2, 0.5)])
    y.append(0)

# Class 1: centered around (4, 4)
for _ in range(50):
    X.append([random.gauss(4, 0.5), random.gauss(4, 0.5)])
    y.append(1)

# Split data (80/20)
indices = list(range(100))
random.shuffle(indices)
train_idx, test_idx = indices[:80], indices[80:]

X_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
X_test = [X[i] for i in test_idx]
y_test = [y[i] for i in test_idx]

# Train and evaluate
model = SimpleKNN(k=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate accuracy
correct = sum(1 for a, b in zip(y_test, y_pred) if a == b)
accuracy = correct / len(y_test)

print(f"Model: k-NN (k=3)")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")
print(f"\\nPredictions vs Actual:")
for i in range(min(10, len(y_test))):
    status = "✓" if y_pred[i] == y_test[i] else "✗"
    print(f"  Sample {i+1}: Predicted={y_pred[i]}, Actual={y_test[i]} {status}")
"""

    result = await python_tool.execute(ctx, code=code, timeout=30.0)
    print(result.content)
    print()


async def run_data_preprocessing():
    """Example: Data preprocessing from scratch."""
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)

    python_tool = PythonTool()
    ctx = ToolContext(cwd="/tmp", agent=None)

    code = """
import random
import statistics

# Create sample data with issues
random.seed(42)
n = 100

ages = [random.randint(18, 70) if random.random() > 0.1 else None for _ in range(n)]
incomes = [random.gauss(50000, 15000) if random.random() > 0.05 else None for _ in range(n)]
categories = [random.choice(['A', 'B', 'C']) for _ in range(n)]

print("Before preprocessing:")
print(f"Samples: {n}")
print(f"Missing ages: {sum(1 for a in ages if a is None)}")
print(f"Missing incomes: {sum(1 for i in incomes if i is None)}")

# 1. Impute missing values with median
valid_ages = [a for a in ages if a is not None]
valid_incomes = [i for i in incomes if i is not None]
age_median = statistics.median(valid_ages)
income_median = statistics.median(valid_incomes)

ages_clean = [a if a is not None else age_median for a in ages]
incomes_clean = [i if i is not None else income_median for i in incomes]

# 2. Encode categorical (A=0, B=1, C=2)
category_map = {'A': 0, 'B': 1, 'C': 2}
categories_encoded = [category_map[c] for c in categories]

# 3. Scale numeric features (z-score)
age_mean, age_std = statistics.mean(ages_clean), statistics.stdev(ages_clean)
income_mean, income_std = statistics.mean(incomes_clean), statistics.stdev(incomes_clean)

ages_scaled = [(a - age_mean) / age_std for a in ages_clean]
incomes_scaled = [(i - income_mean) / income_std for i in incomes_clean]

print("\\nAfter preprocessing:")
print(f"Missing values: 0")
print(f"\\nAge (scaled): mean={statistics.mean(ages_scaled):.2f}, std={statistics.stdev(ages_scaled):.2f}")
print(f"Income (scaled): mean={statistics.mean(incomes_scaled):.2f}, std={statistics.stdev(incomes_scaled):.2f}")

print("\\nFirst 5 samples:")
print(f"{'Age':>10} {'Income':>12} {'Cat':>5} {'Age_s':>8} {'Inc_s':>8}")
print("-" * 50)
for i in range(5):
    print(f"{ages_clean[i]:>10.0f} {incomes_clean[i]:>12,.0f} {categories_encoded[i]:>5} {ages_scaled[i]:>8.2f} {incomes_scaled[i]:>8.2f}")
"""

    result = await python_tool.execute(ctx, code=code, timeout=30.0)
    print(result.content)
    print()


async def run_cross_validation():
    """Example: Cross-validation from scratch."""
    print("=" * 60)
    print("Cross-Validation from Scratch")
    print("=" * 60)

    python_tool = PythonTool()
    ctx = ToolContext(cwd="/tmp", agent=None)

    code = """
import random
import math
from collections import Counter

def k_fold_split(n_samples, n_folds=5):
    \"\"\"Generate k-fold indices.\"\"\"
    indices = list(range(n_samples))
    random.shuffle(indices)
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, test_idx))
    return folds

class NaiveBayes:
    \"\"\"Simple Gaussian Naive Bayes.\"\"\"
    def fit(self, X, y):
        self.classes = list(set(y))
        self.stats = {}
        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i] == c]
            self.stats[c] = {
                'prior': len(X_c) / len(X),
                'means': [sum(f[j] for f in X_c)/len(X_c) for j in range(len(X[0]))],
                'stds': [max(0.01, (sum((f[j]-sum(f[j] for f in X_c)/len(X_c))**2 for f in X_c)/len(X_c))**0.5) for j in range(len(X[0]))]
            }

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        best_class, best_prob = None, -float('inf')
        for c in self.classes:
            prob = math.log(self.stats[c]['prior'])
            for j, val in enumerate(x):
                mean, std = self.stats[c]['means'][j], self.stats[c]['stds'][j]
                prob += -0.5 * ((val - mean) / std) ** 2
            if prob > best_prob:
                best_prob, best_class = prob, c
        return best_class

# Generate data
random.seed(42)
X, y = [], []
for _ in range(100):
    if random.random() < 0.5:
        X.append([random.gauss(2, 1), random.gauss(3, 1)])
        y.append(0)
    else:
        X.append([random.gauss(4, 1), random.gauss(5, 1)])
        y.append(1)

# 5-fold cross-validation
folds = k_fold_split(len(X), n_folds=5)
scores = []

print("5-Fold Cross-Validation:")
print("-" * 40)

for fold_num, (train_idx, test_idx) in enumerate(folds):
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = sum(1 for a, b in zip(y_test, y_pred) if a == b) / len(y_test)
    scores.append(accuracy)
    print(f"Fold {fold_num + 1}: Accuracy = {accuracy:.2%}")

print("-" * 40)
mean_score = sum(scores) / len(scores)
std_score = (sum((s - mean_score)**2 for s in scores) / len(scores)) ** 0.5
print(f"Mean: {mean_score:.2%} (+/- {std_score:.2%})")
"""

    result = await python_tool.execute(ctx, code=code, timeout=60.0)
    print(result.content)
    print()


def main():
    """Run practical DS examples."""
    print("\n" + "=" * 60)
    print("Practical Data Science Workflow Examples")
    print("=" * 60 + "\n")

    # Run examples that work without API
    asyncio.run(run_python_code())
    asyncio.run(run_model_training())
    asyncio.run(run_data_preprocessing())
    asyncio.run(run_cross_validation())

    print("=" * 60)
    print("Agent Prompts (require API key):")
    print("=" * 60)
    print("\n1. Data Exploration:")
    print(EXPLORE_DATA_PROMPT[:200] + "...")
    print("\n2. Feature Engineering:")
    print(FEATURE_ENGINEERING_PROMPT[:200] + "...")
    print("\n3. Model Training:")
    print(TRAIN_MODEL_PROMPT[:200] + "...")
    print("\n4. Hyperparameter Tuning:")
    print(HYPERPARAMETER_PROMPT[:200] + "...")
    print("\nRun with API: ds-agent 'your prompt here'")


if __name__ == "__main__":
    main()
