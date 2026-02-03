"""Ray extension for distributed computing."""

from __future__ import annotations

import json
from typing import Any

from agentic_learn.core.extension import Extension, ExtensionAPI
from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class RayClusterTool(Tool):
    """Manage Ray cluster and distributed computing resources."""

    name = "ray_cluster"
    description = """Manage Ray cluster for distributed computing.

Actions:
- status: Get cluster status and resources
- init: Initialize Ray (local or connect to cluster)
- shutdown: Shutdown Ray
- nodes: List cluster nodes
- resources: Show available resources (CPUs, GPUs, memory)
- dashboard: Get Ray dashboard URL

Use this to:
- Start distributed computing sessions
- Check cluster health and resources
- Scale workloads across nodes

Requires: ray[default] package"""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: status, init, shutdown, nodes, resources, dashboard",
            required=True,
        ),
        ToolParameter(
            name="options",
            type=dict,
            description="Action options (address for init, etc.)",
            required=False,
            default={},
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute Ray cluster action."""
        options = options or {}

        try:
            import ray
        except ImportError:
            return ToolResult(
                tool_call_id="",
                content="Ray not installed. Install with: pip install 'ray[default]'",
                is_error=True,
            )

        action = action.lower()

        try:
            if action == "status":
                return self._get_status(ray)
            elif action == "init":
                return self._init_ray(ray, options)
            elif action == "shutdown":
                return self._shutdown_ray(ray)
            elif action == "nodes":
                return self._get_nodes(ray)
            elif action == "resources":
                return self._get_resources(ray)
            elif action == "dashboard":
                return self._get_dashboard(ray)
            else:
                return ToolResult(
                    tool_call_id="",
                    content=f"Unknown action: {action}",
                    is_error=True,
                )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Ray error: {str(e)}",
                is_error=True,
            )

    def _get_status(self, ray: Any) -> ToolResult:
        """Get Ray cluster status."""
        if not ray.is_initialized():
            return ToolResult(
                tool_call_id="",
                content="Ray is not initialized. Use action='init' to start.",
            )

        try:
            context = ray.get_runtime_context()
            nodes = ray.nodes()

            active_nodes = sum(1 for n in nodes if n.get("Alive", False))
            total_cpus = sum(n.get("Resources", {}).get("CPU", 0) for n in nodes if n.get("Alive"))
            total_gpus = sum(n.get("Resources", {}).get("GPU", 0) for n in nodes if n.get("Alive"))

            return ToolResult(
                tool_call_id="",
                content=f"""Ray Cluster Status:
  Initialized: Yes
  Nodes: {active_nodes} active
  Total CPUs: {total_cpus}
  Total GPUs: {total_gpus}
  Dashboard: {ray.get_dashboard_url() or 'Not available'}""",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Ray initialized but error getting status: {e}",
            )

    def _init_ray(self, ray: Any, options: dict[str, Any]) -> ToolResult:
        """Initialize Ray."""
        if ray.is_initialized():
            return ToolResult(
                tool_call_id="",
                content="Ray is already initialized.",
            )

        address = options.get("address", "auto")
        num_cpus = options.get("num_cpus")
        num_gpus = options.get("num_gpus")

        init_kwargs: dict[str, Any] = {}

        if address != "auto":
            init_kwargs["address"] = address
        if num_cpus is not None:
            init_kwargs["num_cpus"] = num_cpus
        if num_gpus is not None:
            init_kwargs["num_gpus"] = num_gpus

        try:
            ray.init(**init_kwargs)
            dashboard_url = ray.get_dashboard_url()

            return ToolResult(
                tool_call_id="",
                content=f"""Ray initialized successfully.
  Dashboard: {dashboard_url or 'Not available'}
  Resources: {ray.cluster_resources()}""",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Failed to initialize Ray: {e}",
                is_error=True,
            )

    def _shutdown_ray(self, ray: Any) -> ToolResult:
        """Shutdown Ray."""
        if not ray.is_initialized():
            return ToolResult(
                tool_call_id="",
                content="Ray is not initialized.",
            )

        ray.shutdown()
        return ToolResult(
            tool_call_id="",
            content="Ray has been shut down.",
        )

    def _get_nodes(self, ray: Any) -> ToolResult:
        """Get cluster nodes."""
        if not ray.is_initialized():
            return ToolResult(
                tool_call_id="",
                content="Ray is not initialized.",
                is_error=True,
            )

        nodes = ray.nodes()
        lines = ["Ray Cluster Nodes:", "=" * 60, ""]

        for node in nodes:
            status = "Active" if node.get("Alive") else "Dead"
            node_ip = node.get("NodeManagerAddress", "Unknown")
            resources = node.get("Resources", {})

            lines.append(f"Node: {node.get('NodeID', 'Unknown')[:12]}...")
            lines.append(f"  Status: {status}")
            lines.append(f"  Address: {node_ip}")
            lines.append(f"  CPUs: {resources.get('CPU', 0)}")
            lines.append(f"  GPUs: {resources.get('GPU', 0)}")
            lines.append(f"  Memory: {resources.get('memory', 0) / (1024**3):.1f} GB")
            lines.append("")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    def _get_resources(self, ray: Any) -> ToolResult:
        """Get available resources."""
        if not ray.is_initialized():
            return ToolResult(
                tool_call_id="",
                content="Ray is not initialized.",
                is_error=True,
            )

        total = ray.cluster_resources()
        available = ray.available_resources()

        lines = ["Ray Cluster Resources:", "=" * 60, ""]
        lines.append(f"{'Resource':<20} {'Total':>12} {'Available':>12} {'Used':>12}")
        lines.append("-" * 60)

        for resource in sorted(total.keys()):
            total_val = total.get(resource, 0)
            avail_val = available.get(resource, 0)
            used_val = total_val - avail_val

            # Format based on resource type
            if "memory" in resource.lower():
                total_str = f"{total_val / (1024**3):.1f} GB"
                avail_str = f"{avail_val / (1024**3):.1f} GB"
                used_str = f"{used_val / (1024**3):.1f} GB"
            else:
                total_str = f"{total_val:.1f}"
                avail_str = f"{avail_val:.1f}"
                used_str = f"{used_val:.1f}"

            lines.append(f"{resource:<20} {total_str:>12} {avail_str:>12} {used_str:>12}")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    def _get_dashboard(self, ray: Any) -> ToolResult:
        """Get Ray dashboard URL."""
        if not ray.is_initialized():
            return ToolResult(
                tool_call_id="",
                content="Ray is not initialized.",
                is_error=True,
            )

        url = ray.get_dashboard_url()
        if url:
            return ToolResult(
                tool_call_id="",
                content=f"Ray Dashboard: http://{url}",
            )
        else:
            return ToolResult(
                tool_call_id="",
                content="Ray dashboard is not available.",
            )


class RayRunTool(Tool):
    """Run distributed tasks with Ray."""

    name = "ray_run"
    description = """Execute distributed tasks using Ray.

Supports:
- Running Python functions as Ray tasks
- Parallel execution across cluster
- GPU task scheduling

Note: For complex distributed workloads, use the python tool
to write Ray code directly.

Examples:
- Run a function on remote workers
- Parallelize data processing
- Distribute model inference"""

    parameters = [
        ToolParameter(
            name="code",
            type=str,
            description="Python code to run as Ray task",
            required=True,
        ),
        ToolParameter(
            name="num_cpus",
            type=float,
            description="CPUs per task (default: 1)",
            required=False,
            default=1.0,
        ),
        ToolParameter(
            name="num_gpus",
            type=float,
            description="GPUs per task (default: 0)",
            required=False,
            default=0.0,
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        code: str,
        num_cpus: float = 1.0,
        num_gpus: float = 0.0,
    ) -> ToolResult:
        """Run code as Ray task."""
        try:
            import ray
        except ImportError:
            return ToolResult(
                tool_call_id="",
                content="Ray not installed. Install with: pip install 'ray[default]'",
                is_error=True,
            )

        if not ray.is_initialized():
            return ToolResult(
                tool_call_id="",
                content="Ray is not initialized. Use ray_cluster action='init' first.",
                is_error=True,
            )

        # Create a Ray remote function dynamically
        try:
            # Wrap user code in a function
            wrapped_code = f"""
import ray

@ray.remote(num_cpus={num_cpus}, num_gpus={num_gpus})
def _agent_task():
{chr(10).join('    ' + line for line in code.split(chr(10)))}
    return locals().get('result', None)

result = ray.get(_agent_task.remote())
"""
            # Execute
            namespace: dict[str, Any] = {}
            exec(wrapped_code, namespace)
            result = namespace.get("result")

            return ToolResult(
                tool_call_id="",
                content=f"Ray task completed.\nResult: {result}",
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Ray task failed: {str(e)}",
                is_error=True,
            )


class RayExtension(Extension):
    """Extension for Ray distributed computing."""

    name = "ray"
    description = "Distributed computing with Ray"
    version = "0.1.0"

    def setup(self, api: ExtensionAPI) -> None:
        """Register Ray tools."""
        api.register_tool(RayClusterTool())
        api.register_tool(RayRunTool())

        # Register commands
        api.register_command(
            "ray",
            "Ray cluster management: /ray <status|init|shutdown>",
            self._ray_command,
        )

    async def _ray_command(self, ctx: Any, args: list[str]) -> None:
        """Handle /ray command."""
        action = args[0] if args else "status"
        tool = RayClusterTool()
        result = await tool.execute(ctx, action=action)
        print(result.content)
