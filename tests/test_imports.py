"""Smoke tests: verify the package imports and core types are accessible."""

from aglearn import Experiment, Journal, TaskConfig, evolve


def test_exports_exist():
    """All public exports should be importable."""
    assert callable(evolve)
    assert TaskConfig is not None
    assert Journal is not None
    assert Experiment is not None


def test_task_config_creation():
    """TaskConfig should accept required fields."""
    task = TaskConfig(
        description="test task",
        data_path="/tmp/fake.csv",
        target_column="y",
        metric="accuracy",
    )
    assert task.description == "test task"
    assert task.target_column == "y"
