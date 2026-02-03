"""Built-in tools for the ML/DS coding agent.

Tools are organized in tiers:

Tier 1 - Core (always loaded):
    read, write, edit, bash, python
    These are the fundamental coding agent capabilities.

Tier 2 - DS Essentials (loaded by default):
    gpu, data, experiment
    Essential for any ML/DS workflow.

Tier 3 - Advanced (loaded on demand):
    jobs, tune, viz, notebook, repro
    For complex, long-running, or specialized workflows.
"""

from typing import Literal

# Tier 1: Core coding agent tools
from agentic_learn.tools.read_tool import ReadTool
from agentic_learn.tools.write_tool import WriteTool
from agentic_learn.tools.edit_tool import EditTool
from agentic_learn.tools.bash_tool import BashTool
from agentic_learn.tools.python_tool import PythonTool

# Tier 2: DS essentials
from agentic_learn.tools.gpu_tool import GPUTool
from agentic_learn.tools.data_tool import DataTool
from agentic_learn.tools.experiment_tool import ExperimentTool

# Tier 3: Advanced workflows
from agentic_learn.tools.jobs_tool import JobsTool
from agentic_learn.tools.notebook_tool import NotebookTool
from agentic_learn.tools.tuning_tool import TuningTool
from agentic_learn.tools.viz_tool import VizTool
from agentic_learn.tools.repro_tool import ReproTool


__all__ = [
    # Tier 1: Core
    "ReadTool",
    "WriteTool",
    "EditTool",
    "BashTool",
    "PythonTool",
    # Tier 2: DS Essentials
    "GPUTool",
    "DataTool",
    "ExperimentTool",
    # Tier 3: Advanced
    "JobsTool",
    "NotebookTool",
    "TuningTool",
    "VizTool",
    "ReproTool",
    # Tier functions
    "get_tier1_tools",
    "get_tier2_tools",
    "get_tier3_tools",
    "get_tools",
]


# =============================================================================
# Tier Definitions
# =============================================================================

def get_tier1_tools():
    """Tier 1: Core coding agent tools.

    - read: Read files and directories
    - write: Create/overwrite files
    - edit: Find-and-replace editing
    - bash: Execute shell commands
    - python: Execute Python with persistent namespace

    These are the fundamental capabilities every coding agent needs.
    """
    return [
        ReadTool(),
        WriteTool(),
        EditTool(),
        BashTool(),
        PythonTool(),
    ]


def get_tier2_tools():
    """Tier 2: DS essential tools.

    - gpu: Monitor GPU resources, memory, processes
    - data: Load, explore, profile datasets
    - experiment: Track experiments, log metrics

    Essential for any ML/DS workflow. Provides reliable,
    structured output vs raw bash/python commands.
    """
    return [
        GPUTool(),
        DataTool(),
        ExperimentTool(),
    ]


def get_tier3_tools():
    """Tier 3: Advanced workflow tools.

    - jobs: Background jobs with progress tracking, checkpointing
    - tune: Hyperparameter tuning and search
    - viz: Create visualizations
    - notebook: Jupyter notebook manipulation
    - repro: Reproducibility (seeds, snapshots, configs)

    For complex, long-running, or specialized workflows.
    Load these when needed for advanced use cases.
    """
    return [
        JobsTool(),
        TuningTool(),
        VizTool(),
        NotebookTool(),
        ReproTool(),
    ]


# =============================================================================
# Tool Loading Functions
# =============================================================================

Tier = Literal[1, 2, 3, "all"]


def get_tools(tier: Tier = 2) -> list:
    """Get tools up to the specified tier.

    Args:
        tier: Which tiers to include
            1 - Core only (read, write, edit, bash, python)
            2 - Core + DS essentials (default)
            3 - Core + DS essentials + Advanced
            "all" - Same as 3

    Returns:
        List of tool instances

    Examples:
        get_tools(1)     # Minimal: just coding tools
        get_tools(2)     # Default: coding + DS essentials
        get_tools(3)     # Full: all tools
        get_tools("all") # Same as 3
    """
    tools = get_tier1_tools()

    if tier in (2, 3, "all"):
        tools.extend(get_tier2_tools())

    if tier in (3, "all"):
        tools.extend(get_tier3_tools())

    return tools


def get_default_tools():
    """Get default tools (Tier 1 + Tier 2).

    Includes:
    - Core: read, write, edit, bash, python
    - DS Essentials: gpu, data, experiment

    This is the recommended set for most DS coding tasks.
    """
    return get_tools(tier=2)


# =============================================================================
# Tool Info
# =============================================================================

TOOL_INFO = {
    # Tier 1
    "read": {"tier": 1, "description": "Read files and directories"},
    "write": {"tier": 1, "description": "Create and write files"},
    "edit": {"tier": 1, "description": "Edit files with find-replace"},
    "bash": {"tier": 1, "description": "Execute shell commands"},
    "python": {"tier": 1, "description": "Execute Python code"},
    # Tier 2
    "gpu": {"tier": 2, "description": "Monitor GPU resources"},
    "data": {"tier": 2, "description": "Load and explore datasets"},
    "experiment": {"tier": 2, "description": "Track experiments and metrics"},
    # Tier 3
    "jobs": {"tier": 3, "description": "Background job management"},
    "tune": {"tier": 3, "description": "Hyperparameter tuning"},
    "viz": {"tier": 3, "description": "Create visualizations"},
    "notebook": {"tier": 3, "description": "Jupyter notebook manipulation"},
    "repro": {"tier": 3, "description": "Reproducibility management"},
}


def list_tools():
    """List all available tools organized by tier."""
    output = []

    for tier_num in [1, 2, 3]:
        tier_names = {1: "Core", 2: "DS Essentials", 3: "Advanced"}
        output.append(f"\nTier {tier_num}: {tier_names[tier_num]}")
        output.append("-" * 40)

        for name, info in TOOL_INFO.items():
            if info["tier"] == tier_num:
                output.append(f"  {name:<12} {info['description']}")

    return "\n".join(output)
