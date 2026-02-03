"""Built-in tools for the ML/DS agent."""

from agentic_learn.tools.python_tool import PythonTool
from agentic_learn.tools.gpu_tool import GPUTool
from agentic_learn.tools.data_tool import DataTool
from agentic_learn.tools.experiment_tool import ExperimentTool

__all__ = [
    "PythonTool",
    "GPUTool",
    "DataTool",
    "ExperimentTool",
]


def get_default_tools():
    """Get the default set of tools for ML/DS work."""
    return [
        PythonTool(),
        GPUTool(),
        DataTool(),
        ExperimentTool(),
    ]
