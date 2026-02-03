"""GPU monitoring and management tool."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class GPUTool(Tool):
    """Monitor and manage GPU resources.

    Provides information about GPU availability, memory usage,
    running processes, and hardware specifications.
    """

    name = "gpu"
    description = """Monitor GPU resources and get hardware information.

Actions:
- status: Get current GPU status (memory, utilization, temperature)
- list: List all available GPUs with specifications
- processes: Show processes using GPU memory
- memory: Detailed memory breakdown per GPU
- watch: Continuous monitoring (returns single snapshot)

Use this to:
- Check GPU availability before training
- Monitor memory usage during experiments
- Identify GPU bottlenecks
- Find which processes are using GPU resources"""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action to perform: 'status', 'list', 'processes', 'memory', or 'watch'",
            required=True,
        ),
        ToolParameter(
            name="gpu_id",
            type=int,
            description="Specific GPU ID to query (optional, default: all GPUs)",
            required=False,
            default=None,
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        gpu_id: int | None = None,
    ) -> ToolResult:
        """Execute GPU monitoring action."""
        action = action.lower()

        if action == "status":
            return await self._get_status(gpu_id)
        elif action == "list":
            return await self._list_gpus()
        elif action == "processes":
            return await self._get_processes(gpu_id)
        elif action == "memory":
            return await self._get_memory(gpu_id)
        elif action == "watch":
            return await self._get_status(gpu_id)  # Same as status for single snapshot
        else:
            return ToolResult(
                tool_call_id="",
                content=f"Unknown action: {action}. Valid actions: status, list, processes, memory, watch",
                is_error=True,
            )

    async def _run_nvidia_smi(self, args: list[str]) -> tuple[str, bool]:
        """Run nvidia-smi with given arguments."""
        if not shutil.which("nvidia-smi"):
            return "nvidia-smi not found. No NVIDIA GPU available or drivers not installed.", True

        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return f"nvidia-smi error: {stderr.decode()}", True

            return stdout.decode(), False
        except Exception as e:
            return f"Error running nvidia-smi: {e}", True

    async def _get_status(self, gpu_id: int | None = None) -> ToolResult:
        """Get GPU status with memory and utilization."""
        # Try pynvml first for more detailed info
        try:
            return await self._get_status_pynvml(gpu_id)
        except Exception:
            pass

        # Fall back to nvidia-smi
        args = [
            "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        if gpu_id is not None:
            args.extend(["-i", str(gpu_id)])

        output, is_error = await self._run_nvidia_smi(args)

        if is_error:
            return ToolResult(tool_call_id="", content=output, is_error=True)

        # Parse and format output
        lines = output.strip().split("\n")
        formatted = ["GPU Status:", "=" * 60]

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                idx, name, mem_used, mem_total, mem_free, util, temp, power = parts[:8]
                formatted.extend([
                    f"\nGPU {idx}: {name}",
                    f"  Memory: {mem_used} / {mem_total} MiB ({mem_free} MiB free)",
                    f"  Utilization: {util}%",
                    f"  Temperature: {temp}°C",
                    f"  Power: {power}W",
                ])

        return ToolResult(
            tool_call_id="",
            content="\n".join(formatted),
            metadata={"gpu_count": len(lines)},
        )

    async def _get_status_pynvml(self, gpu_id: int | None = None) -> ToolResult:
        """Get GPU status using pynvml for more detail."""
        import pynvml

        pynvml.nvmlInit()

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            formatted = ["GPU Status:", "=" * 60]

            gpu_ids = [gpu_id] if gpu_id is not None else range(device_count)

            for i in gpu_ids:
                if i >= device_count:
                    continue

                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = "N/A"

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                    power_str = f"{power:.1f}W / {power_limit:.1f}W"
                except Exception:
                    power_str = "N/A"

                mem_used_gb = memory.used / (1024**3)
                mem_total_gb = memory.total / (1024**3)
                mem_free_gb = memory.free / (1024**3)
                mem_percent = (memory.used / memory.total) * 100

                formatted.extend([
                    f"\nGPU {i}: {name}",
                    f"  Memory: {mem_used_gb:.1f} / {mem_total_gb:.1f} GB ({mem_percent:.1f}% used, {mem_free_gb:.1f} GB free)",
                    f"  GPU Utilization: {utilization.gpu}%",
                    f"  Memory Bandwidth: {utilization.memory}%",
                    f"  Temperature: {temp}°C",
                    f"  Power: {power_str}",
                ])

            return ToolResult(
                tool_call_id="",
                content="\n".join(formatted),
                metadata={"gpu_count": device_count},
            )

        finally:
            pynvml.nvmlShutdown()

    async def _list_gpus(self) -> ToolResult:
        """List all available GPUs with specifications."""
        args = [
            "--query-gpu=index,name,driver_version,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
        ]

        output, is_error = await self._run_nvidia_smi(args)

        if is_error:
            return ToolResult(tool_call_id="", content=output, is_error=True)

        lines = output.strip().split("\n")
        formatted = ["Available GPUs:", "=" * 60]

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                idx, name, driver, mem, compute, pcie_gen, pcie_width = parts[:7]
                formatted.extend([
                    f"\nGPU {idx}: {name}",
                    f"  Driver: {driver}",
                    f"  Memory: {mem} MiB",
                    f"  Compute Capability: {compute}",
                    f"  PCIe: Gen {pcie_gen} x{pcie_width}",
                ])

        return ToolResult(
            tool_call_id="",
            content="\n".join(formatted),
            metadata={"gpu_count": len(lines)},
        )

    async def _get_processes(self, gpu_id: int | None = None) -> ToolResult:
        """Get processes using GPU memory."""
        args = [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]

        output, is_error = await self._run_nvidia_smi(args)

        if is_error:
            return ToolResult(tool_call_id="", content=output, is_error=True)

        if not output.strip():
            return ToolResult(
                tool_call_id="",
                content="No processes currently using GPU compute.",
            )

        lines = output.strip().split("\n")
        formatted = ["GPU Processes:", "=" * 60, ""]
        formatted.append(f"{'PID':<10} {'Memory':<12} {'Process'}")
        formatted.append("-" * 60)

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_uuid, pid, name, mem = parts[:4]
                # Truncate long process names
                if len(name) > 40:
                    name = name[:37] + "..."
                formatted.append(f"{pid:<10} {mem + ' MiB':<12} {name}")

        return ToolResult(
            tool_call_id="",
            content="\n".join(formatted),
            metadata={"process_count": len(lines)},
        )

    async def _get_memory(self, gpu_id: int | None = None) -> ToolResult:
        """Get detailed memory breakdown."""
        # Use pynvml if available for more detail
        try:
            return await self._get_memory_pynvml(gpu_id)
        except Exception:
            pass

        # Fall back to nvidia-smi
        args = [
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        if gpu_id is not None:
            args.extend(["-i", str(gpu_id)])

        output, is_error = await self._run_nvidia_smi(args)

        if is_error:
            return ToolResult(tool_call_id="", content=output, is_error=True)

        lines = output.strip().split("\n")
        formatted = ["GPU Memory:", "=" * 60]

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, total, used, free = parts[:5]
                total_f = float(total)
                used_f = float(used)
                percent = (used_f / total_f) * 100 if total_f > 0 else 0

                # Create visual bar
                bar_width = 40
                filled = int(bar_width * percent / 100)
                bar = "█" * filled + "░" * (bar_width - filled)

                formatted.extend([
                    f"\nGPU {idx}: {name}",
                    f"  [{bar}] {percent:.1f}%",
                    f"  Used:  {used} MiB",
                    f"  Free:  {free} MiB",
                    f"  Total: {total} MiB",
                ])

        return ToolResult(
            tool_call_id="",
            content="\n".join(formatted),
        )

    async def _get_memory_pynvml(self, gpu_id: int | None = None) -> ToolResult:
        """Get detailed memory using pynvml."""
        import pynvml

        pynvml.nvmlInit()

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            formatted = ["GPU Memory:", "=" * 60]

            gpu_ids = [gpu_id] if gpu_id is not None else range(device_count)

            for i in gpu_ids:
                if i >= device_count:
                    continue

                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                total_gb = memory.total / (1024**3)
                used_gb = memory.used / (1024**3)
                free_gb = memory.free / (1024**3)
                percent = (memory.used / memory.total) * 100

                bar_width = 40
                filled = int(bar_width * percent / 100)
                bar = "█" * filled + "░" * (bar_width - filled)

                formatted.extend([
                    f"\nGPU {i}: {name}",
                    f"  [{bar}] {percent:.1f}%",
                    f"  Used:  {used_gb:.2f} GB",
                    f"  Free:  {free_gb:.2f} GB",
                    f"  Total: {total_gb:.2f} GB",
                ])

            return ToolResult(
                tool_call_id="",
                content="\n".join(formatted),
            )

        finally:
            pynvml.nvmlShutdown()
