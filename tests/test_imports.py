"""Smoke tests: verify the package imports and core types are accessible."""

from aglearn import Experiment, Journal, TaskConfig, evolve
from aglearn.data.synth import SyntheticTask
from aglearn.runtime.agent import AgentCLIConfig, claude_cli_config, run
from aglearn.runtime.loop import EvaluationResult, TaskConfig as RuntimeTaskConfig
from aglearn.storage.journal import Journal as StorageJournal
from aglearn_experiments import BenchmarkManifest


def test_exports_exist():
    assert callable(evolve)
    assert TaskConfig is not None
    assert Journal is not None
    assert Experiment is not None


def test_task_config_creation():
    task = TaskConfig(
        description="test task",
        data_path="/tmp/fake.csv",
        target_column="y",
        metric="accuracy",
    )
    assert task.description == "test task"
    assert task.target_column == "y"


def test_reorganized_subpackages_are_importable():
    assert run is not None
    assert AgentCLIConfig is not None
    assert EvaluationResult is not None
    assert claude_cli_config is not None
    assert RuntimeTaskConfig is not None
    assert StorageJournal is not None
    assert SyntheticTask is not None
    assert BenchmarkManifest is not None
