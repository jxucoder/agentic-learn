"""Modal integration for parallel MLE-bench development.

Uses Modal for:
1. Parallel LLM calls across multiple solution branches
2. Parallel evaluation of competitions during development
3. Running the full Lite split (22 competitions) concurrently

NOTE: For official MLE-bench leaderboard submission, the agent runs inside
the MLE-bench Docker container. Modal is for development speed only.

Usage:
    # Run a single competition on Modal
    modal run src/modal_runner.py --competition spaceship-titanic

    # Run the Lite split in parallel
    modal run src/modal_runner.py --split lite
"""

from __future__ import annotations

import os
from typing import Optional

# Modal imports are optional -- this module degrades gracefully
try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False


def check_modal_available() -> bool:
    """Check if Modal is available and configured."""
    return HAS_MODAL


# =============================================================================
# Modal App Definition (only created if Modal is available)
# =============================================================================

if HAS_MODAL:
    app = modal.App("mlebench-agent")

    # Image with our dependencies
    agent_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "httpx>=0.27.0",
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
    )

    @app.function(
        image=agent_image,
        timeout=3600,
        secrets=[
            modal.Secret.from_name("anthropic-secret", required=False),
            modal.Secret.from_name("openai-secret", required=False),
        ],
    )
    def generate_solution(
        task_description: str,
        task_context: str,
        approach: str,
        provider: str = "anthropic",
        model: Optional[str] = None,
    ) -> dict:
        """Generate a solution for a competition on Modal.

        Args:
            task_description: Competition description text.
            task_context: Parsed task context string.
            approach: Approach to try (e.g., "xgboost_baseline").
            provider: LLM provider.
            model: LLM model name.

        Returns:
            Dict with 'code' and 'metadata'.
        """
        from src.llm import LLMClient
        from src.search import SEARCH_SYSTEM_PROMPT

        llm = LLMClient(provider=provider, model=model)

        prompt = f"""## Competition
{task_description[:4000]}

## Task Summary
{task_context[:2000]}

## Approach: {approach}

Write a complete Python solution implementing this approach.
Save submission to /home/submission/submission.csv.
Print the CV score.
"""

        code = llm.generate_code(prompt, system=SEARCH_SYSTEM_PROMPT)

        return {
            "approach": approach,
            "code": code,
            "code_length": len(code),
        }

    @app.function(
        image=agent_image,
        timeout=7200,
        secrets=[
            modal.Secret.from_name("anthropic-secret", required=False),
            modal.Secret.from_name("openai-secret", required=False),
        ],
    )
    def generate_solutions_batch(
        task_description: str,
        task_context: str,
        approaches: list[str],
        provider: str = "anthropic",
        model: Optional[str] = None,
    ) -> list[dict]:
        """Generate multiple solutions in parallel on Modal.

        Args:
            task_description: Competition description.
            task_context: Parsed task context.
            approaches: List of approaches to try.
            provider: LLM provider.
            model: LLM model name.

        Returns:
            List of dicts with 'approach', 'code', 'metadata'.
        """
        results = []
        for approach in approaches:
            result = generate_solution.remote(
                task_description=task_description,
                task_context=task_context,
                approach=approach,
                provider=provider,
                model=model,
            )
            results.append(result)

        return results

    @app.local_entrypoint()
    def main(
        competition: str = "spaceship-titanic",
        provider: str = "anthropic",
        model: Optional[str] = None,
    ):
        """Run solution generation for a competition."""
        print(f"Generating solutions for: {competition}")
        print(f"Provider: {provider}, Model: {model}")

        # For local testing, generate a few approaches
        approaches = ["xgboost_baseline", "lightgbm_baseline", "random_forest"]

        # This would use the task parser in a real setup
        task_desc = f"Competition: {competition}"
        task_ctx = f"Task: {competition}"

        results = generate_solutions_batch.remote(
            task_description=task_desc,
            task_context=task_ctx,
            approaches=approaches,
            provider=provider,
            model=model,
        )

        for r in results:
            print(f"\n--- {r['approach']} ---")
            print(f"Code length: {r['code_length']} chars")
            print(r['code'][:200] + "...")


# =============================================================================
# Non-Modal Fallback
# =============================================================================

def generate_solutions_local(
    task_description: str,
    task_context: str,
    approaches: list[str],
    provider: str = "anthropic",
    model: Optional[str] = None,
) -> list[dict]:
    """Generate solutions locally (sequential, no Modal).

    Fallback for when Modal is not available.
    """
    from src.llm import LLMClient
    from src.search import SEARCH_SYSTEM_PROMPT

    llm = LLMClient(provider=provider, model=model)
    results = []

    for approach in approaches:
        prompt = f"""## Competition
{task_description[:4000]}

## Task Summary
{task_context[:2000]}

## Approach: {approach}

Write a complete Python solution implementing this approach.
Save submission to /home/submission/submission.csv.
Print the CV score.
"""
        code = llm.generate_code(prompt, system=SEARCH_SYSTEM_PROMPT)
        results.append({
            "approach": approach,
            "code": code,
            "code_length": len(code),
        })

    return results
