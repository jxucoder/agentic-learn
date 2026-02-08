"""MCTS search for MLE-bench solution space exploration.

Adapts Silver's MCTS (PUCT selection, backpropagation) to search over
complete Python solution scripts for Kaggle competitions.

Each node in the tree represents a complete solution attempt:
- Root: task description
- Depth 1: different approach families (XGBoost, LightGBM, NN, ...)
- Depth 2+: variations/improvements on each approach

The search uses PUCT to balance exploration of new approaches
vs deepening the best-performing branches.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from src.budget import BudgetManager
from src.llm import LLMClient
from src.runner import ExecutionResult, execute_script
from src.task_parser import TaskInfo
from src.validator import validate_submission

logger = logging.getLogger(__name__)


# =============================================================================
# Node
# =============================================================================

class NodeState(Enum):
    """State of a search node."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TERMINAL = "terminal"


@dataclass
class SearchNode:
    """A node in the solution search tree.

    Each node is a complete Python solution script with its execution results.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Tree structure
    parent: Optional[SearchNode] = None
    children: list[SearchNode] = field(default_factory=list)
    depth: int = 0

    # Solution content
    approach: str = ""          # e.g., "xgboost_baseline", "lightgbm_tuned"
    description: str = ""       # Natural language description of what this tries
    code: str = ""              # Full Python script

    # Execution results
    state: NodeState = NodeState.PENDING
    score: Optional[float] = None
    execution_time: float = 0.0
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    submission_valid: bool = False

    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    prior: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def q_value(self) -> float:
        """Average reward."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def puct_score(self, c: float = 1.414) -> float:
        """PUCT score for node selection."""
        if self.parent is None:
            return 0.0
        exploitation = self.q_value
        exploration = c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration

    def compute_reward(self, higher_is_better: bool = True) -> float:
        """Compute reward in [-1, 1] from execution results."""
        if self.state == NodeState.FAILED:
            return -1.0
        if self.score is None:
            if self.submission_valid:
                return -0.3  # Valid submission but no score
            return -0.5
        # Normalize to [-1, 1]
        if higher_is_better:
            reward = 2 * min(self.score, 1.0) - 1
        else:
            reward = 1 - 2 * min(self.score, 1.0)
        return max(-1.0, min(1.0, reward))

    def update(self, reward: float) -> None:
        """Update MCTS statistics."""
        self.visits += 1
        self.total_reward += reward

    def add_child(self, child: SearchNode) -> None:
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def summary(self) -> str:
        """Brief summary for logging."""
        status = "OK" if self.state == NodeState.SUCCESS else "FAIL" if self.state == NodeState.FAILED else "?"
        score_str = f"{self.score:.4f}" if self.score is not None else "N/A"
        return f"[{self.id}] {status} {self.approach}: score={score_str} visits={self.visits} Q={self.q_value:.3f}"


# =============================================================================
# Search Config
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for MCTS search."""
    max_iterations: int = 100
    exploration_constant: float = 1.414
    max_depth: int = 5
    max_children_per_node: int = 4
    patience: int = 15           # Stop if no improvement for N iterations
    debug_prob: float = 1.0      # Probability of debugging a failed node
    max_debug_depth: int = 3     # Max attempts to fix a failing approach


# =============================================================================
# MCTS Search Engine
# =============================================================================

SEARCH_SYSTEM_PROMPT = """You are an expert ML engineer generating solution code for a Kaggle competition.

Write a COMPLETE, SELF-CONTAINED Python script that:
1. Loads training and test data
2. Preprocesses data
3. Trains a model
4. Makes predictions on the test set
5. Saves submission.csv to the specified path
6. Prints the cross-validation score

Return ONLY the Python code inside a ```python code block.
The script must be runnable end-to-end with `python solution.py`.
"""


class MCTSSearch:
    """MCTS search over the solution space for an MLE-bench competition.

    Usage:
        search = MCTSSearch(task, llm, config=config)
        result = search.run(budget=budget, code_dir="/home/code", submission_dir="/home/submission")
    """

    def __init__(
        self,
        task: TaskInfo,
        llm: LLMClient,
        config: Optional[SearchConfig] = None,
    ):
        self.task = task
        self.llm = llm
        self.config = config or SearchConfig()

        # Tree
        self.root = SearchNode(
            id="root",
            approach="root",
            description="Root node",
            state=NodeState.SUCCESS,
            prior=1.0,
            visits=1,
        )
        self.nodes: dict[str, SearchNode] = {self.root.id: self.root}
        self.best_node: Optional[SearchNode] = None
        self.best_score: Optional[float] = None

        # Tracking
        self.iterations = 0
        self.no_improvement_count = 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(
        self,
        budget: BudgetManager,
        code_dir: str = "/home/code",
        submission_dir: str = "/home/submission",
        step_timeout: float = 32400,
    ) -> SearchResult:
        """Run the MCTS search.

        Args:
            budget: Time budget manager.
            code_dir: Directory for solution scripts.
            submission_dir: Directory for submissions.
            step_timeout: Max time per script execution.

        Returns:
            SearchResult with the best solution found.
        """
        start_time = time.time()

        # Initial expansion: generate approach variants from root
        self._initial_expand()

        # Main search loop
        while not self._should_stop(budget):
            self.iterations += 1
            logger.info(f"--- MCTS Iteration {self.iterations} | {budget.status()} ---")

            # 1. Select a node to work on
            node = self._select(self.root)

            # 2. If leaf and not at max depth, expand
            if node.is_leaf and node.depth < self.config.max_depth and node.state == NodeState.SUCCESS:
                children = self._expand(node)
                if children:
                    node = children[0]

            # 3. If pending, evaluate (generate code + execute)
            if node.state == NodeState.PENDING:
                self._evaluate(node, code_dir, submission_dir, budget, step_timeout)

            # 4. Backpropagate
            reward = node.compute_reward(higher_is_better=self.task.higher_is_better)
            self._backpropagate(node, reward)

            # Track best
            self._update_best(node, submission_dir)

            # Log
            logger.info(f"  Node: {node.summary()}")
            logger.info(f"  Tree: {len(self.nodes)} nodes, best={self.best_score}")

        # Compile results
        elapsed = time.time() - start_time
        result = SearchResult(
            best_node=self.best_node,
            best_score=self.best_score,
            best_code=self.best_node.code if self.best_node else "",
            total_iterations=self.iterations,
            total_time=elapsed,
            nodes_explored=len(self.nodes),
        )

        logger.info(f"MCTS search complete: {result.total_iterations} iterations, "
                     f"best_score={result.best_score}, nodes={result.nodes_explored}")

        return result

    # -----------------------------------------------------------------
    # MCTS Phases
    # -----------------------------------------------------------------

    def _select(self, node: SearchNode) -> SearchNode:
        """Select a leaf node using PUCT."""
        while not node.is_leaf:
            if node.depth >= self.config.max_depth:
                return node
            best_score = float("-inf")
            best_child = node.children[0]
            for child in node.children:
                s = child.puct_score(self.config.exploration_constant)
                if s > best_score:
                    best_score = s
                    best_child = child
            node = best_child
        return node

    def _initial_expand(self) -> None:
        """Expand root with initial approach variants."""
        approaches = self._get_initial_approaches()
        n = len(approaches)
        for name, desc in approaches:
            child = SearchNode(
                approach=name,
                description=desc,
                prior=1.0 / n,
                state=NodeState.PENDING,
            )
            self.root.add_child(child)
            self.nodes[child.id] = child

        logger.info(f"Initial expansion: {n} approaches")

    def _expand(self, parent: SearchNode) -> list[SearchNode]:
        """Expand a successful node with variation/improvement children."""
        # Generate variations based on the parent's approach and results
        variations = self._generate_variations(parent)
        children = []
        n = len(variations)
        for name, desc in variations[:self.config.max_children_per_node]:
            child = SearchNode(
                approach=name,
                description=desc,
                prior=1.0 / max(n, 1),
                state=NodeState.PENDING,
            )
            parent.add_child(child)
            self.nodes[child.id] = child
            children.append(child)

        return children

    def _evaluate(
        self,
        node: SearchNode,
        code_dir: str,
        submission_dir: str,
        budget: BudgetManager,
        step_timeout: float,
    ) -> None:
        """Evaluate a node: generate code and execute it."""
        node.state = NodeState.RUNNING

        # Generate code
        try:
            prompt = self._build_node_prompt(node)
            code = self.llm.generate_code(prompt, system=SEARCH_SYSTEM_PROMPT)
            node.code = code
        except Exception as e:
            node.state = NodeState.FAILED
            node.error_message = f"Code generation failed: {e}"
            logger.warning(f"Node {node.id} code gen failed: {e}")
            return

        if not code:
            node.state = NodeState.FAILED
            node.error_message = "Empty code generated"
            return

        # Execute
        timeout = budget.step_timeout(max_timeout=step_timeout)
        result = execute_script(
            code=code,
            code_dir=code_dir,
            submission_dir=submission_dir,
            timeout=timeout,
            iteration=self.iterations,
            script_name=f"solution_{node.id}.py",
        )

        # Record results
        node.stdout = result.stdout
        node.stderr = result.stderr
        node.execution_time = result.execution_time_seconds

        if result.success and result.submission_created:
            # Validate
            val = validate_submission(submission_path=result.submission_path or f"{submission_dir}/submission.csv")
            node.submission_valid = val.valid
            node.score = result.score
            node.state = NodeState.SUCCESS
        else:
            node.state = NodeState.FAILED
            node.error_message = result.error_message or result.stderr[:500] or "Unknown error"

    def _backpropagate(self, node: SearchNode, reward: float) -> None:
        """Backpropagate reward from node to root."""
        current: Optional[SearchNode] = node
        while current is not None:
            current.update(reward)
            current = current.parent

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _should_stop(self, budget: BudgetManager) -> bool:
        """Check if search should stop."""
        if budget.should_stop():
            return True
        if self.iterations >= self.config.max_iterations:
            return True
        if self.no_improvement_count >= self.config.patience:
            logger.info("Stopping: no improvement (patience exceeded)")
            return True
        return False

    def _update_best(self, node: SearchNode, submission_dir: str) -> None:
        """Update best node tracking."""
        if node.state != NodeState.SUCCESS or node.score is None:
            self.no_improvement_count += 1
            return

        is_better = False
        if self.best_score is None:
            is_better = True
        elif self.task.higher_is_better and node.score > self.best_score:
            is_better = True
        elif not self.task.higher_is_better and node.score < self.best_score:
            is_better = True

        if is_better:
            self.best_node = node
            self.best_score = node.score
            self.no_improvement_count = 0

            # Save best submission as backup
            submission_csv = Path(submission_dir) / "submission.csv"
            best_backup = Path(submission_dir) / "best_submission.csv"
            if submission_csv.exists():
                shutil.copy2(str(submission_csv), str(best_backup))

            logger.info(f"NEW BEST: {node.approach} score={node.score:.6f}")
        else:
            self.no_improvement_count += 1

    def _get_initial_approaches(self) -> list[tuple[str, str]]:
        """Get initial approach variants based on task type."""
        task_type = self.task.task_type.value

        # Core approaches that work for most tasks
        approaches = [
            ("xgboost_baseline", f"XGBoost baseline for {task_type}"),
            ("lightgbm_baseline", f"LightGBM baseline for {task_type}"),
            ("random_forest", f"Random Forest baseline for {task_type}"),
        ]

        if "image" in task_type:
            approaches = [
                ("resnet_transfer", "ResNet50 transfer learning"),
                ("efficientnet_transfer", "EfficientNet transfer learning"),
                ("simple_cnn", "Simple CNN from scratch"),
            ]
        elif "text" in task_type:
            approaches = [
                ("tfidf_logreg", "TF-IDF + Logistic Regression"),
                ("tfidf_lgbm", "TF-IDF + LightGBM"),
                ("transformer_finetune", "Transformer fine-tuning"),
            ]
        elif "audio" in task_type:
            approaches = [
                ("spectrogram_cnn", "Mel spectrogram + CNN"),
                ("mfcc_xgboost", "MFCC features + XGBoost"),
            ]

        return approaches

    def _generate_variations(self, parent: SearchNode) -> list[tuple[str, str]]:
        """Generate variation approaches from a successful parent node."""
        base = parent.approach
        return [
            (f"{base}_tuned", f"Hyperparameter tuned version of {base}"),
            (f"{base}_feat_eng", f"{base} with advanced feature engineering"),
            (f"{base}_ensemble", f"Ensemble combining {base} with other models"),
        ]

    def _build_node_prompt(self, node: SearchNode) -> str:
        """Build the LLM prompt for a search node."""
        parts = [
            f"## Competition: {self.task.competition_id}",
            f"Metric: {self.task.metric_name} ({'higher is better' if self.task.higher_is_better else 'lower is better'})",
            "",
            f"## Approach: {node.approach}",
            f"Description: {node.description}",
            "",
            "## Competition Description (excerpt):",
            self.task.description[:4000],
            "",
            "## Data Summary:",
            self.task.to_context(),
            "",
            f"## Save submission to: {self.task.submission_dir}/submission.csv",
            f"## Data directory: {self.task.data_dir}",
        ]

        # Add parent context if available
        if node.parent and node.parent.code and node.parent.state == NodeState.SUCCESS:
            parts.extend([
                "",
                "## Parent Solution (to improve upon):",
                f"Parent approach: {node.parent.approach}",
                f"Parent score: {node.parent.score}",
                f"Parent code (excerpt):",
                f"```python",
                node.parent.code[-4000:],
                "```",
                "",
                f"Your task: Write a variation that improves on this. Approach: {node.description}",
            ])
        else:
            parts.extend([
                "",
                f"Write a complete Python solution implementing: {node.description}",
                "Print the CV score clearly (e.g., 'CV Score: 0.85').",
            ])

        return "\n".join(parts)


# =============================================================================
# Result
# =============================================================================

@dataclass
class SearchResult:
    """Result of MCTS search."""
    best_node: Optional[SearchNode] = None
    best_score: Optional[float] = None
    best_code: str = ""
    total_iterations: int = 0
    total_time: float = 0.0
    nodes_explored: int = 0
