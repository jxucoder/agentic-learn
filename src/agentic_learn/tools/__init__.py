"""Built-in tools for the ML/DS agent."""

from agentic_learn.tools.python_tool import PythonTool
from agentic_learn.tools.gpu_tool import GPUTool
from agentic_learn.tools.data_tool import DataTool
from agentic_learn.tools.experiment_tool import ExperimentTool
from agentic_learn.tools.jobs_tool import JobsTool
from agentic_learn.tools.notebook_tool import NotebookTool
from agentic_learn.tools.tuning_tool import TuningTool
from agentic_learn.tools.viz_tool import VizTool
from agentic_learn.tools.repro_tool import ReproTool

__all__ = [
    # Core tools
    "PythonTool",
    "GPUTool",
    "DataTool",
    "ExperimentTool",
    # DS-specific tools
    "JobsTool",
    "NotebookTool",
    "TuningTool",
    "VizTool",
    "ReproTool",
]


def get_default_tools():
    """Get the default set of tools for ML/DS work.

    Core tools (always loaded):
    - python: Execute Python code with persistent namespace
    - gpu: Monitor GPU resources
    - data: Load and explore datasets
    - experiment: Track experiments and metrics

    DS-specific tools:
    - jobs: Background job management for long-running tasks
    - notebook: Jupyter notebook creation and manipulation
    - tune: Hyperparameter tuning
    - viz: Visualization creation
    - repro: Reproducibility management
    """
    return [
        # Core
        PythonTool(),
        GPUTool(),
        DataTool(),
        ExperimentTool(),
        # DS-specific
        JobsTool(),
        NotebookTool(),
        TuningTool(),
        VizTool(),
        ReproTool(),
    ]


def get_core_tools():
    """Get only the core tools (minimal set)."""
    return [
        PythonTool(),
        GPUTool(),
        DataTool(),
        ExperimentTool(),
    ]


def get_ds_tools():
    """Get DS-specific tools (for long-running/advanced workflows)."""
    return [
        JobsTool(),
        NotebookTool(),
        TuningTool(),
        VizTool(),
        ReproTool(),
    ]
