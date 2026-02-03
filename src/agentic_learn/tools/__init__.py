"""Built-in tools for the ML/DS coding agent."""

# Coding agent core tools (like pi-mono)
from agentic_learn.tools.read_tool import ReadTool
from agentic_learn.tools.write_tool import WriteTool
from agentic_learn.tools.edit_tool import EditTool
from agentic_learn.tools.bash_tool import BashTool

# DS-specific tools
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
    # Coding agent core (file operations)
    "ReadTool",
    "WriteTool",
    "EditTool",
    "BashTool",
    # Code execution
    "PythonTool",
    # DS-specific
    "GPUTool",
    "DataTool",
    "ExperimentTool",
    "JobsTool",
    "NotebookTool",
    "TuningTool",
    "VizTool",
    "ReproTool",
]


def get_default_tools():
    """Get the default set of tools for the DS coding agent.

    Coding agent core (like pi-mono):
    - read: Read files
    - write: Write/create files
    - edit: Edit files (find/replace)
    - bash: Run shell commands

    DS-specific tools:
    - python: Execute Python code with persistent namespace
    - gpu: Monitor GPU resources
    - data: Load and explore datasets
    - experiment: Track experiments and metrics
    - jobs: Background job management
    - notebook: Jupyter notebook manipulation
    - tune: Hyperparameter tuning
    - viz: Visualization creation
    - repro: Reproducibility management
    """
    return [
        # Coding agent core
        ReadTool(),
        WriteTool(),
        EditTool(),
        BashTool(),
        # Code execution
        PythonTool(),
        # DS-specific
        GPUTool(),
        DataTool(),
        ExperimentTool(),
        JobsTool(),
        NotebookTool(),
        TuningTool(),
        VizTool(),
        ReproTool(),
    ]


def get_coding_tools():
    """Get just the coding agent tools (minimal set for code writing)."""
    return [
        ReadTool(),
        WriteTool(),
        EditTool(),
        BashTool(),
    ]


def get_ds_tools():
    """Get DS-specific tools (for ML/data science workflows)."""
    return [
        PythonTool(),
        GPUTool(),
        DataTool(),
        ExperimentTool(),
        JobsTool(),
        NotebookTool(),
        TuningTool(),
        VizTool(),
        ReproTool(),
    ]
