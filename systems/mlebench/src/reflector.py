"""Structured reflector for diagnosing failures and extracting learnings.

After each execution, the reflector:
1. Diagnoses failures (syntax error? runtime error? low score? bad submission?)
2. Extracts learnings ("feature X caused overfitting", "need more epochs")
3. Suggests next action: fix, improve, or try new approach
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

from src.llm import LLMClient
from src.runner import ExecutionResult

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Categories of failures."""
    NONE = "none"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    DATA_ERROR = "data_error"           # File not found, wrong columns, etc.
    MEMORY_ERROR = "memory_error"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    BAD_SUBMISSION = "bad_submission"    # Submission format wrong
    LOW_SCORE = "low_score"
    NO_SUBMISSION = "no_submission"      # Code ran but no submission produced


class Action(Enum):
    """Suggested next action."""
    FIX_ERROR = "fix_error"           # Fix the specific error
    IMPROVE = "improve"               # Improve the current approach
    TRY_NEW = "try_new"               # Try a completely different approach
    SIMPLIFY = "simplify"             # Simplify the approach
    ENSEMBLE = "ensemble"             # Combine existing solutions


@dataclass
class Reflection:
    """Result of reflection on an execution."""
    failure_type: FailureType = FailureType.NONE
    diagnosis: str = ""
    learnings: list[str] = field(default_factory=list)
    suggested_action: Action = Action.IMPROVE
    action_detail: str = ""
    confidence: float = 0.5


class Reflector:
    """Analyzes execution results and produces actionable reflection.

    Can work in two modes:
    1. Rule-based: fast, deterministic diagnosis from error patterns
    2. LLM-assisted: deeper analysis using the LLM (optional)
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def reflect(
        self,
        result: ExecutionResult,
        code: str,
        score_history: list[Optional[float]] | None = None,
        best_score: Optional[float] = None,
    ) -> Reflection:
        """Reflect on an execution result.

        Args:
            result: The execution result.
            code: The code that was executed.
            score_history: Previous scores for trend analysis.
            best_score: Best score achieved so far.

        Returns:
            Reflection with diagnosis, learnings, and suggested action.
        """
        reflection = Reflection()

        # Step 1: Diagnose failure type
        reflection.failure_type = self._diagnose_failure(result)

        # Step 2: Generate diagnosis message
        reflection.diagnosis = self._generate_diagnosis(result, reflection.failure_type)

        # Step 3: Extract learnings
        reflection.learnings = self._extract_learnings(result, code, reflection.failure_type)

        # Step 4: Suggest action
        reflection.suggested_action, reflection.action_detail = self._suggest_action(
            reflection.failure_type, result, score_history, best_score
        )

        logger.info(
            f"Reflection: {reflection.failure_type.value} -> "
            f"{reflection.suggested_action.value}: {reflection.action_detail[:100]}"
        )

        return reflection

    def _diagnose_failure(self, result: ExecutionResult) -> FailureType:
        """Diagnose the type of failure from execution result."""
        if result.timed_out:
            return FailureType.TIMEOUT

        if result.success and result.submission_created:
            if result.score is None:
                return FailureType.NONE  # Ran OK, just no score parsed
            return FailureType.NONE

        if result.success and not result.submission_created:
            return FailureType.NO_SUBMISSION

        # Analyze error
        stderr = (result.stderr or "").lower()
        error_type = (result.error_type or "").lower()
        error_msg = (result.error_message or "").lower()

        if "syntaxerror" in error_type or "syntaxerror" in stderr:
            return FailureType.SYNTAX_ERROR

        if "importerror" in error_type or "modulenotfounderror" in error_type:
            return FailureType.IMPORT_ERROR

        if "filenotfounderror" in error_type or "no such file" in stderr:
            return FailureType.DATA_ERROR

        if "keyerror" in error_type or "column" in error_msg:
            return FailureType.DATA_ERROR

        if "memoryerror" in error_type or "killed" in stderr:
            return FailureType.MEMORY_ERROR

        return FailureType.RUNTIME_ERROR

    def _generate_diagnosis(self, result: ExecutionResult, failure_type: FailureType) -> str:
        """Generate a human-readable diagnosis."""
        if failure_type == FailureType.NONE:
            if result.score is not None:
                return f"Success with score {result.score:.6f}"
            return "Execution succeeded"

        messages = {
            FailureType.SYNTAX_ERROR: "Python syntax error in generated code",
            FailureType.IMPORT_ERROR: f"Missing package: {result.error_message or 'unknown'}",
            FailureType.DATA_ERROR: f"Data issue: {result.error_message or 'file/column not found'}",
            FailureType.MEMORY_ERROR: "Out of memory -- dataset too large or model too big",
            FailureType.TIMEOUT: f"Execution timed out after {result.execution_time_seconds:.0f}s",
            FailureType.RUNTIME_ERROR: f"Runtime error: {result.error_type}: {result.error_message or ''}",
            FailureType.NO_SUBMISSION: "Code ran but did not produce submission.csv",
            FailureType.BAD_SUBMISSION: "Submission file has wrong format",
        }

        return messages.get(failure_type, f"Unknown failure: {result.error_type}")

    def _extract_learnings(
        self, result: ExecutionResult, code: str, failure_type: FailureType
    ) -> list[str]:
        """Extract actionable learnings from the result."""
        learnings = []

        if failure_type == FailureType.IMPORT_ERROR:
            # Identify which package is missing
            for pkg in ("xgboost", "lightgbm", "catboost", "torch", "tensorflow",
                        "transformers", "sklearn", "cv2", "librosa"):
                if pkg in (result.error_message or "").lower():
                    learnings.append(f"Package '{pkg}' may not be installed -- try alternatives")

        elif failure_type == FailureType.DATA_ERROR:
            learnings.append("Check data file paths and column names carefully")
            learnings.append("Read data files before assuming column structure")

        elif failure_type == FailureType.MEMORY_ERROR:
            learnings.append("Reduce data size: sample rows or reduce features")
            learnings.append("Use lighter models (e.g., LightGBM instead of deep learning)")

        elif failure_type == FailureType.TIMEOUT:
            learnings.append("Reduce training epochs or use early stopping")
            learnings.append("Use simpler model or smaller dataset sample")

        elif failure_type == FailureType.NO_SUBMISSION:
            learnings.append("Make sure to save submission.csv to the correct path")
            learnings.append("Check that the prediction step completes without error")

        elif failure_type == FailureType.NONE and result.score is not None:
            # Successful run -- extract score-related learnings
            if result.score < 0.5:
                learnings.append("Score is low -- consider different features or model")
            elif result.score > 0.9:
                learnings.append("Very high score -- check for data leakage")

        # Check stdout for warnings
        if result.stdout:
            if "convergence" in result.stdout.lower():
                learnings.append("Model convergence warning -- increase iterations or adjust learning rate")
            if "class imbalance" in result.stdout.lower() or "imbalanced" in result.stdout.lower():
                learnings.append("Class imbalance detected -- consider class weights or SMOTE")

        return learnings

    def _suggest_action(
        self,
        failure_type: FailureType,
        result: ExecutionResult,
        score_history: list[Optional[float]] | None,
        best_score: Optional[float],
    ) -> tuple[Action, str]:
        """Suggest the next action based on failure analysis."""
        if failure_type in (FailureType.SYNTAX_ERROR, FailureType.IMPORT_ERROR,
                            FailureType.DATA_ERROR, FailureType.NO_SUBMISSION):
            return Action.FIX_ERROR, f"Fix the {failure_type.value}: {result.error_message or ''}"

        if failure_type == FailureType.MEMORY_ERROR:
            return Action.SIMPLIFY, "Reduce model complexity or data size"

        if failure_type == FailureType.TIMEOUT:
            return Action.SIMPLIFY, "Use faster training (fewer epochs, simpler model, early stopping)"

        if failure_type == FailureType.RUNTIME_ERROR:
            return Action.FIX_ERROR, f"Fix runtime error: {result.error_type}: {result.error_message}"

        # Success case -- decide whether to improve or try new
        if failure_type == FailureType.NONE:
            # Check for score stagnation
            if score_history and len(score_history) >= 3:
                recent = [s for s in score_history[-3:] if s is not None]
                if len(recent) >= 2 and all(abs(recent[-1] - s) < 0.001 for s in recent[:-1]):
                    return Action.TRY_NEW, "Score has stagnated -- try a different approach or ensemble"

            # Check if score is already good
            if result.score is not None and result.score > 0.85:
                return Action.ENSEMBLE, "Score is good -- try ensembling with other approaches"

            return Action.IMPROVE, "Improve current approach with better features or hyperparameters"

        return Action.FIX_ERROR, "Fix the issue and retry"


def reflect_with_llm(
    llm: LLMClient,
    result: ExecutionResult,
    code: str,
    task_context: str,
) -> Reflection:
    """Use LLM for deeper reflection (optional, more expensive).

    Args:
        llm: LLM client.
        result: Execution result.
        code: The code that was executed.
        task_context: Task description context.

    Returns:
        Reflection with LLM-powered analysis.
    """
    prompt = f"""Analyze this ML solution attempt and provide reflection.

## Task Context
{task_context[:2000]}

## Code (excerpt):
```python
{code[-3000:]}
```

## Execution Result:
- Success: {result.success}
- Score: {result.score}
- Error: {result.error_type}: {result.error_message}
- stdout (last 500 chars): {result.stdout[-500:]}
- stderr (last 500 chars): {result.stderr[-500:]}

## Provide:
1. DIAGNOSIS: What went wrong (or right)?
2. LEARNINGS: 2-3 key takeaways (one per line, prefix with "- ")
3. NEXT ACTION: What should we try next?

Be concise and actionable.
"""

    response = llm.complete(prompt, temperature=0.3, max_tokens=1000)
    content = response.content

    # Parse response
    reflection = Reflection()

    # Extract learnings (lines starting with "- ")
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            reflection.learnings.append(stripped[2:])

    reflection.diagnosis = content[:500]
    reflection.confidence = 0.7

    return reflection
