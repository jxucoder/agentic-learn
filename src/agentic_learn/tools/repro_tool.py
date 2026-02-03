"""Reproducibility tool for ML experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class ReproTool(Tool):
    """Ensure reproducibility in ML experiments."""

    name = "repro"
    description = """Manage reproducibility for ML experiments.

Actions:
- seed: Set random seeds for all frameworks (numpy, torch, tensorflow, random)
- snapshot: Capture current environment state (packages, git, system)
- restore: Restore environment from snapshot
- config: Save/load experiment configuration
- env: Show current environment info
- diff: Compare two snapshots
- freeze: Create requirements.txt from current environment
- hash: Compute hash of data/model files

Features:
- Consistent seeding across all ML frameworks
- Environment snapshots with package versions
- Git commit tracking
- Configuration management
- Data/model checksums

Example workflow:
1. seed value=42  # Set all random seeds
2. snapshot name="baseline"  # Save environment state
3. config save name="exp1" data='{"lr": 0.001, "epochs": 100}'
4. ... run experiment ...
5. hash path="model.pt"  # Checksum for verification

Later, to reproduce:
1. restore name="baseline"  # Restore environment
2. config load name="exp1"  # Load settings
3. seed value=42  # Same seed"""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: seed, snapshot, restore, config, env, diff, freeze, hash",
            required=True,
        ),
        ToolParameter(
            name="options",
            type=dict,
            description="Action-specific options",
            required=False,
            default={},
        ),
    ]

    def __init__(self, storage_dir: str = ".ds-agent/repro"):
        super().__init__()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute reproducibility action."""
        options = options or {}
        action = action.lower()

        try:
            if action == "seed":
                return self._set_seeds(options)
            elif action == "snapshot":
                return self._snapshot(options)
            elif action == "restore":
                return self._restore(options)
            elif action == "config":
                return self._config(options)
            elif action == "env":
                return self._env()
            elif action == "diff":
                return self._diff(options)
            elif action == "freeze":
                return self._freeze(options)
            elif action == "hash":
                return self._hash(options)
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

    def _set_seeds(self, options: dict[str, Any]) -> ToolResult:
        """Set random seeds for all frameworks."""
        seed = options.get("value", options.get("seed", 42))
        deterministic = options.get("deterministic", True)

        set_frameworks = []

        # Python random
        import random
        random.seed(seed)
        set_frameworks.append("random")

        # NumPy
        try:
            import numpy as np
            np.random.seed(seed)
            set_frameworks.append("numpy")
        except ImportError:
            pass

        # PyTorch
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if deterministic:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            set_frameworks.append("torch")
        except ImportError:
            pass

        # TensorFlow
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            set_frameworks.append("tensorflow")
        except ImportError:
            pass

        # Set PYTHONHASHSEED environment variable
        os.environ["PYTHONHASHSEED"] = str(seed)
        set_frameworks.append("PYTHONHASHSEED")

        return ToolResult(
            tool_call_id="",
            content=f"""Seeds set to {seed}:
  Frameworks: {', '.join(set_frameworks)}
  Deterministic: {deterministic}

Note: For full reproducibility, also set CUBLAS_WORKSPACE_CONFIG=:4096:8""",
            metadata={"seed": seed, "frameworks": set_frameworks},
        )

    def _snapshot(self, options: dict[str, Any]) -> ToolResult:
        """Create environment snapshot."""
        name = options.get("name", datetime.now().strftime("%Y%m%d_%H%M%S"))

        snapshot = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "system": self._get_system_info(),
            "python": self._get_python_info(),
            "packages": self._get_packages(),
            "git": self._get_git_info(),
            "env_vars": self._get_relevant_env_vars(),
        }

        # Save snapshot
        path = self.storage_dir / "snapshots" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

        return ToolResult(
            tool_call_id="",
            content=f"""Environment snapshot saved: {name}
  Python: {snapshot['python']['version']}
  Packages: {len(snapshot['packages'])}
  Git: {snapshot['git'].get('commit', 'N/A')[:8] if snapshot['git'] else 'N/A'}
  Path: {path}""",
            metadata={"path": str(path), "name": name},
        )

    def _get_system_info(self) -> dict[str, str]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
        }

    def _get_python_info(self) -> dict[str, str]:
        """Get Python information."""
        return {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        }

    def _get_packages(self) -> dict[str, str]:
        """Get installed packages."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
            )
            packages = {}
            for line in result.stdout.strip().split("\n"):
                if "==" in line:
                    name, version = line.split("==", 1)
                    packages[name.lower()] = version
            return packages
        except Exception:
            return {}

    def _get_git_info(self) -> dict[str, str] | None:
        """Get git repository information."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode != 0:
                return None

            commit = result.stdout.strip()

            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            )
            branch = result.stdout.strip()

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
            )
            dirty = bool(result.stdout.strip())

            return {
                "commit": commit,
                "branch": branch,
                "dirty": dirty,
            }
        except Exception:
            return None

    def _get_relevant_env_vars(self) -> dict[str, str]:
        """Get relevant environment variables."""
        relevant = [
            "PYTHONPATH",
            "PYTHONHASHSEED",
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ]
        return {k: v for k, v in os.environ.items() if k in relevant}

    def _restore(self, options: dict[str, Any]) -> ToolResult:
        """Restore environment from snapshot."""
        name = options.get("name")
        if not name:
            return ToolResult(
                tool_call_id="",
                content="Snapshot name required.",
                is_error=True,
            )

        path = self.storage_dir / "snapshots" / f"{name}.json"
        if not path.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Snapshot not found: {name}",
                is_error=True,
            )

        with open(path) as f:
            snapshot = json.load(f)

        # Generate pip install command
        packages = snapshot.get("packages", {})
        requirements = [f"{pkg}=={ver}" for pkg, ver in packages.items()]

        # Check git status
        git_info = snapshot.get("git")
        git_cmd = ""
        if git_info:
            git_cmd = f"\n# Restore git state:\ngit checkout {git_info['commit']}"

        # Set env vars
        env_vars = snapshot.get("env_vars", {})
        env_cmd = "\n".join(f"export {k}={v}" for k, v in env_vars.items())

        return ToolResult(
            tool_call_id="",
            content=f"""To restore environment '{name}':

# 1. Install packages:
pip install {' '.join(requirements[:10])}{'...' if len(requirements) > 10 else ''}

# Full requirements saved to:
{self.storage_dir / 'snapshots' / f'{name}_requirements.txt'}

# 2. Set environment variables:
{env_cmd if env_cmd else '# (none)'}
{git_cmd}

Snapshot from: {snapshot.get('created_at', 'Unknown')}""",
        )

    def _config(self, options: dict[str, Any]) -> ToolResult:
        """Save or load experiment configuration."""
        action = options.get("action", "save" if options.get("data") else "load")
        name = options.get("name", "default")

        config_dir = self.storage_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{name}.json"

        if action == "save":
            data = options.get("data", {})
            if isinstance(data, str):
                data = json.loads(data)

            # Add metadata
            config = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "config": data,
            }

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            return ToolResult(
                tool_call_id="",
                content=f"Configuration saved: {config_path}",
                metadata={"path": str(config_path)},
            )

        elif action == "load":
            if not config_path.exists():
                return ToolResult(
                    tool_call_id="",
                    content=f"Configuration not found: {name}",
                    is_error=True,
                )

            with open(config_path) as f:
                config = json.load(f)

            return ToolResult(
                tool_call_id="",
                content=f"""Configuration '{name}':
{json.dumps(config.get('config', {}), indent=2)}

Created: {config.get('created_at', 'Unknown')}""",
                metadata={"config": config.get("config", {})},
            )

        elif action == "list":
            configs = list(config_dir.glob("*.json"))
            if not configs:
                return ToolResult(
                    tool_call_id="",
                    content="No configurations found.",
                )

            lines = ["Saved configurations:", ""]
            for cfg in sorted(configs):
                with open(cfg) as f:
                    data = json.load(f)
                created = data.get("created_at", "Unknown")[:19]
                lines.append(f"  {cfg.stem}: {created}")

            return ToolResult(
                tool_call_id="",
                content="\n".join(lines),
            )

        else:
            return ToolResult(
                tool_call_id="",
                content=f"Unknown config action: {action}. Use save, load, or list.",
                is_error=True,
            )

    def _env(self) -> ToolResult:
        """Show current environment information."""
        system = self._get_system_info()
        python = self._get_python_info()
        git = self._get_git_info()

        # Check for ML frameworks
        frameworks = []
        try:
            import numpy as np
            frameworks.append(f"numpy {np.__version__}")
        except ImportError:
            pass

        try:
            import torch
            cuda = f" (CUDA {torch.version.cuda})" if torch.cuda.is_available() else ""
            frameworks.append(f"torch {torch.__version__}{cuda}")
        except ImportError:
            pass

        try:
            import tensorflow as tf
            frameworks.append(f"tensorflow {tf.__version__}")
        except ImportError:
            pass

        try:
            import sklearn
            frameworks.append(f"sklearn {sklearn.__version__}")
        except ImportError:
            pass

        git_str = ""
        if git:
            git_str = f"""
Git:
  Commit: {git['commit'][:8]}
  Branch: {git['branch']}
  Dirty: {git['dirty']}"""

        return ToolResult(
            tool_call_id="",
            content=f"""Environment Info:

System:
  Platform: {system['platform']}
  Machine: {system['machine']}

Python:
  Version: {python['version']}
  Executable: {python['executable']}

ML Frameworks:
  {chr(10).join('  ' + f for f in frameworks) if frameworks else '  (none detected)'}
{git_str}""",
        )

    def _diff(self, options: dict[str, Any]) -> ToolResult:
        """Compare two snapshots."""
        snap1 = options.get("snapshot1") or options.get("a")
        snap2 = options.get("snapshot2") or options.get("b")

        if not snap1 or not snap2:
            return ToolResult(
                tool_call_id="",
                content="Two snapshot names required (snapshot1, snapshot2 or a, b).",
                is_error=True,
            )

        path1 = self.storage_dir / "snapshots" / f"{snap1}.json"
        path2 = self.storage_dir / "snapshots" / f"{snap2}.json"

        if not path1.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Snapshot not found: {snap1}",
                is_error=True,
            )
        if not path2.exists():
            return ToolResult(
                tool_call_id="",
                content=f"Snapshot not found: {snap2}",
                is_error=True,
            )

        with open(path1) as f:
            data1 = json.load(f)
        with open(path2) as f:
            data2 = json.load(f)

        # Compare packages
        pkg1 = set(data1.get("packages", {}).items())
        pkg2 = set(data2.get("packages", {}).items())

        added = {k: v for k, v in (pkg2 - pkg1)}
        removed = {k: v for k, v in (pkg1 - pkg2)}
        changed = {}

        for pkg, ver1 in data1.get("packages", {}).items():
            ver2 = data2.get("packages", {}).get(pkg)
            if ver2 and ver1 != ver2:
                changed[pkg] = (ver1, ver2)

        lines = [f"Diff: {snap1} → {snap2}", "=" * 50, ""]

        if added:
            lines.append("Added packages:")
            for pkg, ver in list(added.items())[:10]:
                lines.append(f"  + {pkg}=={ver}")
            if len(added) > 10:
                lines.append(f"  ... and {len(added) - 10} more")
            lines.append("")

        if removed:
            lines.append("Removed packages:")
            for pkg, ver in list(removed.items())[:10]:
                lines.append(f"  - {pkg}=={ver}")
            if len(removed) > 10:
                lines.append(f"  ... and {len(removed) - 10} more")
            lines.append("")

        if changed:
            lines.append("Changed packages:")
            for pkg, (v1, v2) in list(changed.items())[:10]:
                lines.append(f"  ~ {pkg}: {v1} → {v2}")
            if len(changed) > 10:
                lines.append(f"  ... and {len(changed) - 10} more")

        if not added and not removed and not changed:
            lines.append("No differences in packages.")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    def _freeze(self, options: dict[str, Any]) -> ToolResult:
        """Create requirements.txt."""
        output = options.get("output", "requirements.txt")
        path = Path(output)

        packages = self._get_packages()

        with open(path, "w") as f:
            for pkg, ver in sorted(packages.items()):
                f.write(f"{pkg}=={ver}\n")

        return ToolResult(
            tool_call_id="",
            content=f"Requirements written to {path} ({len(packages)} packages)",
        )

    def _hash(self, options: dict[str, Any]) -> ToolResult:
        """Compute file hash."""
        path = options.get("path")
        if not path:
            return ToolResult(
                tool_call_id="",
                content="File path required.",
                is_error=True,
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                tool_call_id="",
                content=f"File not found: {path}",
                is_error=True,
            )

        # Compute SHA256 hash
        sha256 = hashlib.sha256()
        with open(path_obj, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        file_hash = sha256.hexdigest()
        file_size = path_obj.stat().st_size

        return ToolResult(
            tool_call_id="",
            content=f"""File: {path}
SHA256: {file_hash}
Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)""",
            metadata={"hash": file_hash, "size": file_size},
        )
