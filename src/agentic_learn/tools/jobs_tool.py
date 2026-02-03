"""Background job management for long-running tasks."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class JobStatus(str, Enum):
    """Status of a background job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for a job."""
    current: int = 0
    total: int = 100
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0


@dataclass
class Job:
    """A background job."""
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: JobProgress = field(default_factory=JobProgress)
    result: Any = None
    error: str | None = None
    checkpoint_path: str | None = None
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": {
                "current": self.progress.current,
                "total": self.progress.total,
                "percent": self.progress.percent,
                "message": self.progress.message,
                "metrics": self.progress.metrics,
            },
            "error": self.error,
            "checkpoint_path": self.checkpoint_path,
        }


class JobManager:
    """Manages background jobs with progress tracking and checkpointing."""

    def __init__(self, storage_dir: str = ".ds-agent/jobs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._processes: dict[str, mp.Process] = {}
        self._progress_queues: dict[str, mp.Queue] = {}
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load existing jobs from disk."""
        for job_file in self.storage_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    data = json.load(f)
                    job = Job(
                        id=data["id"],
                        name=data["name"],
                        status=JobStatus(data["status"]),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                        error=data.get("error"),
                        checkpoint_path=data.get("checkpoint_path"),
                    )
                    # Mark running jobs as failed (process died)
                    if job.status == JobStatus.RUNNING:
                        job.status = JobStatus.FAILED
                        job.error = "Process terminated unexpectedly"
                    self.jobs[job.id] = job
            except Exception:
                pass

    def _save_job(self, job: Job) -> None:
        """Save job to disk."""
        job_file = self.storage_dir / f"{job.id}.json"
        with open(job_file, "w") as f:
            json.dump(job.to_dict(), f, indent=2)

    def _generate_id(self) -> str:
        """Generate a unique job ID."""
        import hashlib
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]

    def create_job(self, name: str) -> Job:
        """Create a new job."""
        job = Job(id=self._generate_id(), name=name)
        self.jobs[job.id] = job
        self._save_job(job)
        return job

    def submit(
        self,
        job: Job,
        func: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> None:
        """Submit a job for background execution."""
        kwargs = kwargs or {}

        # Create progress queue
        progress_queue: mp.Queue = mp.Queue()
        self._progress_queues[job.id] = progress_queue

        # Create checkpoint directory
        checkpoint_dir = self.storage_dir / "checkpoints" / job.id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        job.checkpoint_path = str(checkpoint_dir)

        # Start process
        process = mp.Process(
            target=self._run_job,
            args=(job.id, func, args, kwargs, progress_queue, str(checkpoint_dir)),
        )
        self._processes[job.id] = process

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self._save_job(job)

        process.start()

        # Start progress monitor thread
        threading.Thread(
            target=self._monitor_progress,
            args=(job.id,),
            daemon=True,
        ).start()

    @staticmethod
    def _run_job(
        job_id: str,
        func: Callable,
        args: tuple,
        kwargs: dict,
        progress_queue: mp.Queue,
        checkpoint_dir: str,
    ) -> None:
        """Run a job in a separate process."""
        # Create a progress reporter for the function
        class ProgressReporter:
            def __init__(self, queue: mp.Queue, ckpt_dir: str):
                self.queue = queue
                self.checkpoint_dir = Path(ckpt_dir)

            def update(
                self,
                current: int | None = None,
                total: int | None = None,
                message: str = "",
                **metrics: float,
            ) -> None:
                self.queue.put({
                    "type": "progress",
                    "current": current,
                    "total": total,
                    "message": message,
                    "metrics": metrics,
                })

            def log(self, message: str) -> None:
                self.queue.put({"type": "log", "message": message})

            def checkpoint(self, state: Any, name: str = "checkpoint") -> str:
                """Save a checkpoint."""
                path = self.checkpoint_dir / f"{name}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(state, f)
                self.queue.put({"type": "checkpoint", "path": str(path)})
                return str(path)

            def load_checkpoint(self, name: str = "checkpoint") -> Any | None:
                """Load a checkpoint if it exists."""
                path = self.checkpoint_dir / f"{name}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        return pickle.load(f)
                return None

        reporter = ProgressReporter(progress_queue, checkpoint_dir)

        try:
            result = func(*args, progress=reporter, **kwargs)
            progress_queue.put({"type": "complete", "result": result})
        except Exception as e:
            progress_queue.put({
                "type": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    def _monitor_progress(self, job_id: str) -> None:
        """Monitor job progress from the queue."""
        job = self.jobs.get(job_id)
        queue = self._progress_queues.get(job_id)

        if not job or not queue:
            return

        while True:
            try:
                msg = queue.get(timeout=1.0)

                if msg["type"] == "progress":
                    if msg.get("current") is not None:
                        job.progress.current = msg["current"]
                    if msg.get("total") is not None:
                        job.progress.total = msg["total"]
                    if msg.get("message"):
                        job.progress.message = msg["message"]
                    if msg.get("metrics"):
                        job.progress.metrics.update(msg["metrics"])
                    self._save_job(job)

                elif msg["type"] == "log":
                    job.logs.append(msg["message"])

                elif msg["type"] == "checkpoint":
                    job.checkpoint_path = msg["path"]
                    self._save_job(job)

                elif msg["type"] == "complete":
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    job.result = msg.get("result")
                    job.progress.current = job.progress.total
                    job.progress.message = "Completed"
                    self._save_job(job)
                    break

                elif msg["type"] == "error":
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now()
                    job.error = msg["error"]
                    self._save_job(job)
                    break

            except Exception:
                # Check if process is still alive
                process = self._processes.get(job_id)
                if process and not process.is_alive():
                    if job.status == JobStatus.RUNNING:
                        job.status = JobStatus.FAILED
                        job.error = "Process terminated unexpectedly"
                        job.completed_at = datetime.now()
                        self._save_job(job)
                    break

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        process = self._processes.get(job_id)

        if not job or job.status != JobStatus.RUNNING:
            return False

        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        self._save_job(job)
        return True

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self, status: JobStatus | None = None) -> list[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)


# Global job manager instance
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


class JobsTool(Tool):
    """Manage long-running background jobs."""

    name = "jobs"
    description = """Manage long-running background jobs with progress tracking and checkpointing.

Actions:
- submit: Submit a training/processing job to run in background
- status: Check job status and progress
- list: List all jobs
- logs: View job logs
- cancel: Cancel a running job
- resume: Resume a job from checkpoint

Use this for:
- Model training that takes hours
- Large data processing pipelines
- Hyperparameter sweeps
- Any task you don't want to wait for

Jobs support:
- Progress tracking with metrics
- Automatic checkpointing
- Resume from failure
- Cancellation

Example flow:
1. submit code="train_model(epochs=100)" name="resnet-training"
2. status job_id="abc123" (check progress)
3. The job runs in background, saves checkpoints
4. If interrupted: resume job_id="abc123\""""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: submit, status, list, logs, cancel, resume",
            required=True,
        ),
        ToolParameter(
            name="options",
            type=dict,
            description="Action options (code, name, job_id, etc.)",
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
        """Execute job action."""
        options = options or {}
        manager = get_job_manager()
        action = action.lower()

        try:
            if action == "submit":
                return await self._submit(manager, ctx, options)
            elif action == "status":
                return self._status(manager, options)
            elif action == "list":
                return self._list(manager, options)
            elif action == "logs":
                return self._logs(manager, options)
            elif action == "cancel":
                return self._cancel(manager, options)
            elif action == "resume":
                return await self._resume(manager, ctx, options)
            else:
                return ToolResult(
                    tool_call_id="",
                    content=f"Unknown action: {action}",
                    is_error=True,
                )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Error: {str(e)}",
                is_error=True,
            )

    async def _submit(
        self,
        manager: JobManager,
        ctx: ToolContext,
        options: dict[str, Any],
    ) -> ToolResult:
        """Submit a new job."""
        code = options.get("code")
        name = options.get("name", f"job-{datetime.now().strftime('%H%M%S')}")

        if not code:
            return ToolResult(
                tool_call_id="",
                content="Code is required for job submission",
                is_error=True,
            )

        # Create job
        job = manager.create_job(name)

        # Create a function that executes the code
        def run_code(progress):
            # Set up namespace with common imports and progress reporter
            namespace = {
                "progress": progress,
                "checkpoint": progress.checkpoint,
                "load_checkpoint": progress.load_checkpoint,
            }

            # Common imports
            setup = """
import numpy as np
import time
try:
    import pandas as pd
except: pass
try:
    import torch
except: pass
try:
    import tensorflow as tf
except: pass
"""
            exec(setup, namespace)
            exec(code, namespace)
            return namespace.get("result")

        # Submit job
        manager.submit(job, run_code)

        return ToolResult(
            tool_call_id="",
            content=f"""Job submitted:
  ID: {job.id}
  Name: {job.name}
  Status: {job.status.value}

Use 'status job_id="{job.id}"' to check progress.
Use 'cancel job_id="{job.id}"' to stop the job.""",
            metadata={"job_id": job.id},
        )

    def _status(self, manager: JobManager, options: dict[str, Any]) -> ToolResult:
        """Get job status."""
        job_id = options.get("job_id")

        if not job_id:
            return ToolResult(
                tool_call_id="",
                content="job_id is required",
                is_error=True,
            )

        job = manager.get_job(job_id)
        if not job:
            return ToolResult(
                tool_call_id="",
                content=f"Job not found: {job_id}",
                is_error=True,
            )

        # Create progress bar
        bar_width = 30
        filled = int(bar_width * job.progress.percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        duration = ""
        if job.started_at:
            end = job.completed_at or datetime.now()
            duration = str(end - job.started_at).split(".")[0]

        metrics_str = ""
        if job.progress.metrics:
            metrics_str = "\n  Metrics: " + ", ".join(
                f"{k}={v:.4f}" for k, v in job.progress.metrics.items()
            )

        return ToolResult(
            tool_call_id="",
            content=f"""Job: {job.name} ({job.id})
  Status: {job.status.value}
  Progress: [{bar}] {job.progress.percent:.1f}%
  Message: {job.progress.message or 'N/A'}
  Duration: {duration or 'Not started'}{metrics_str}
  Checkpoint: {job.checkpoint_path or 'None'}
  Error: {job.error or 'None'}""",
        )

    def _list(self, manager: JobManager, options: dict[str, Any]) -> ToolResult:
        """List all jobs."""
        status_filter = options.get("status")
        status = JobStatus(status_filter) if status_filter else None

        jobs = manager.list_jobs(status)

        if not jobs:
            return ToolResult(
                tool_call_id="",
                content="No jobs found.",
            )

        lines = ["Jobs:", "=" * 70, ""]
        lines.append(f"{'ID':<12} {'Name':<20} {'Status':<12} {'Progress':<10} {'Created'}")
        lines.append("-" * 70)

        for job in jobs[:20]:
            created = job.created_at.strftime("%m-%d %H:%M")
            progress = f"{job.progress.percent:.0f}%"
            lines.append(
                f"{job.id:<12} {job.name[:19]:<20} {job.status.value:<12} {progress:<10} {created}"
            )

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    def _logs(self, manager: JobManager, options: dict[str, Any]) -> ToolResult:
        """Get job logs."""
        job_id = options.get("job_id")

        if not job_id:
            return ToolResult(
                tool_call_id="",
                content="job_id is required",
                is_error=True,
            )

        job = manager.get_job(job_id)
        if not job:
            return ToolResult(
                tool_call_id="",
                content=f"Job not found: {job_id}",
                is_error=True,
            )

        if not job.logs:
            return ToolResult(
                tool_call_id="",
                content=f"No logs for job {job_id}",
            )

        # Get last N logs
        n = options.get("n", 50)
        logs = job.logs[-n:]

        return ToolResult(
            tool_call_id="",
            content=f"Logs for {job.name} ({job_id}):\n" + "\n".join(logs),
        )

    def _cancel(self, manager: JobManager, options: dict[str, Any]) -> ToolResult:
        """Cancel a job."""
        job_id = options.get("job_id")

        if not job_id:
            return ToolResult(
                tool_call_id="",
                content="job_id is required",
                is_error=True,
            )

        if manager.cancel(job_id):
            return ToolResult(
                tool_call_id="",
                content=f"Job {job_id} cancelled.",
            )
        else:
            return ToolResult(
                tool_call_id="",
                content=f"Could not cancel job {job_id} (not running or not found)",
                is_error=True,
            )

    async def _resume(
        self,
        manager: JobManager,
        ctx: ToolContext,
        options: dict[str, Any],
    ) -> ToolResult:
        """Resume a job from checkpoint."""
        job_id = options.get("job_id")

        if not job_id:
            return ToolResult(
                tool_call_id="",
                content="job_id is required",
                is_error=True,
            )

        job = manager.get_job(job_id)
        if not job:
            return ToolResult(
                tool_call_id="",
                content=f"Job not found: {job_id}",
                is_error=True,
            )

        if not job.checkpoint_path:
            return ToolResult(
                tool_call_id="",
                content=f"No checkpoint found for job {job_id}",
                is_error=True,
            )

        return ToolResult(
            tool_call_id="",
            content=f"""To resume job {job_id}, load the checkpoint in your code:

checkpoint_path = "{job.checkpoint_path}"
# Load with pickle or your framework's load function

The checkpoint contains the state saved by the original job.""",
        )
