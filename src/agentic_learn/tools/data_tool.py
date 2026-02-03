"""Data loading and exploration tool."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class DataTool(Tool):
    """Load, explore, and profile datasets.

    Supports local files and popular data sources like
    Hugging Face and Kaggle.
    """

    name = "data"
    description = """Load and explore datasets from various sources.

Actions:
- load: Load a dataset from file or source (csv, parquet, json, huggingface, kaggle)
- info: Get basic info about a loaded dataset (shape, columns, dtypes)
- head: Show first N rows
- describe: Statistical summary of numeric columns
- profile: Generate comprehensive data profile
- sample: Random sample of rows
- columns: List all columns with types
- missing: Analyze missing values
- unique: Show unique value counts for columns

Sources:
- Local files: CSV, Parquet, JSON, Excel
- Hugging Face: hf://dataset_name or hf://org/dataset
- Kaggle: kaggle://competition/dataset

Examples:
- load "train.csv"
- load "hf://imdb" split="train"
- info "train.csv"
- describe "data.parquet" columns="age,income\""""

    parameters = [
        ToolParameter(
            name="action",
            type=str,
            description="Action: load, info, head, describe, profile, sample, columns, missing, unique",
            required=True,
        ),
        ToolParameter(
            name="source",
            type=str,
            description="Data source: file path, hf://dataset, or kaggle://dataset",
            required=True,
        ),
        ToolParameter(
            name="options",
            type=dict,
            description="Additional options (n_rows, columns, split, etc.)",
            required=False,
            default={},
        ),
    ]

    def __init__(self):
        super().__init__()
        # Cache for loaded datasets
        self._cache: dict[str, Any] = {}

    async def execute(
        self,
        ctx: ToolContext,
        action: str,
        source: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute data action."""
        options = options or {}
        action = action.lower()

        try:
            if action == "load":
                return await self._load(source, options)
            elif action == "info":
                return await self._info(source, options)
            elif action == "head":
                return await self._head(source, options)
            elif action == "describe":
                return await self._describe(source, options)
            elif action == "profile":
                return await self._profile(source, options)
            elif action == "sample":
                return await self._sample(source, options)
            elif action == "columns":
                return await self._columns(source, options)
            elif action == "missing":
                return await self._missing(source, options)
            elif action == "unique":
                return await self._unique(source, options)
            else:
                return ToolResult(
                    tool_call_id="",
                    content=f"Unknown action: {action}. Valid actions: load, info, head, describe, profile, sample, columns, missing, unique",
                    is_error=True,
                )
        except ImportError as e:
            return ToolResult(
                tool_call_id="",
                content=f"Missing dependency: {e}. Install with: pip install pandas",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Error: {str(e)}",
                is_error=True,
            )

    def _get_dataframe(self, source: str) -> Any:
        """Get a pandas DataFrame from source (from cache or load)."""
        import pandas as pd

        if source in self._cache:
            return self._cache[source]

        # Determine source type and load
        if source.startswith("hf://"):
            df = self._load_huggingface(source[5:])
        elif source.startswith("kaggle://"):
            df = self._load_kaggle(source[9:])
        else:
            # Local file
            df = self._load_file(source)

        self._cache[source] = df
        return df

    def _load_file(self, path: str) -> Any:
        """Load a local file into a DataFrame."""
        import pandas as pd

        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        elif suffix == ".json":
            return pd.read_json(path)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif suffix == ".feather":
            return pd.read_feather(path)
        elif suffix == ".pkl":
            return pd.read_pickle(path)
        else:
            # Try CSV as default
            return pd.read_csv(path)

    def _load_huggingface(self, dataset_name: str, split: str = "train") -> Any:
        """Load a dataset from Hugging Face."""
        import pandas as pd

        try:
            from datasets import load_dataset

            ds = load_dataset(dataset_name, split=split)
            return ds.to_pandas()
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

    def _load_kaggle(self, dataset_path: str) -> Any:
        """Load a dataset from Kaggle."""
        raise NotImplementedError("Kaggle loading not yet implemented. Download manually and use local file.")

    async def _load(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Load a dataset and cache it."""
        import pandas as pd

        # Handle split option for HuggingFace
        if source.startswith("hf://") and "split" in options:
            # Load with specific split
            from datasets import load_dataset
            ds = load_dataset(source[5:], split=options["split"])
            df = ds.to_pandas()
            self._cache[source] = df
        else:
            df = self._get_dataframe(source)

        # Apply row limit if specified
        n_rows = options.get("n_rows")
        if n_rows:
            df = df.head(n_rows)
            self._cache[source] = df

        shape = df.shape
        columns = list(df.columns)
        dtypes = df.dtypes.astype(str).to_dict()
        memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

        return ToolResult(
            tool_call_id="",
            content=f"""Dataset loaded: {source}
Shape: {shape[0]:,} rows × {shape[1]} columns
Memory: {memory:.2f} MB

Columns:
{json.dumps(dtypes, indent=2)}""",
            metadata={"shape": shape, "columns": columns},
        )

    async def _info(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Get dataset info."""
        import io

        df = self._get_dataframe(source)

        # Capture df.info() output
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        return ToolResult(
            tool_call_id="",
            content=f"Dataset Info: {source}\n{'=' * 50}\n{info_str}",
        )

    async def _head(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Show first N rows."""
        df = self._get_dataframe(source)
        n = options.get("n", 5)

        # Select specific columns if requested
        columns = options.get("columns")
        if columns:
            if isinstance(columns, str):
                columns = [c.strip() for c in columns.split(",")]
            df = df[columns]

        head_str = df.head(n).to_string()

        return ToolResult(
            tool_call_id="",
            content=f"First {n} rows of {source}:\n{head_str}",
        )

    async def _describe(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Statistical summary."""
        df = self._get_dataframe(source)

        # Select specific columns if requested
        columns = options.get("columns")
        if columns:
            if isinstance(columns, str):
                columns = [c.strip() for c in columns.split(",")]
            df = df[columns]

        # Include all types
        desc = df.describe(include="all").to_string()

        return ToolResult(
            tool_call_id="",
            content=f"Statistical Summary: {source}\n{'=' * 50}\n{desc}",
        )

    async def _profile(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Comprehensive data profile."""
        import pandas as pd

        df = self._get_dataframe(source)

        profile_parts = [
            f"Data Profile: {source}",
            "=" * 60,
            "",
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
            "",
            "Column Analysis:",
            "-" * 60,
        ]

        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100
            unique = df[col].nunique()

            profile_parts.append(f"\n{col} ({dtype}):")
            profile_parts.append(f"  Non-null: {non_null:,} ({100-null_pct:.1f}%)")
            profile_parts.append(f"  Missing: {null_count:,} ({null_pct:.1f}%)")
            profile_parts.append(f"  Unique: {unique:,}")

            # Type-specific stats
            if pd.api.types.is_numeric_dtype(dtype):
                profile_parts.append(f"  Min: {df[col].min()}")
                profile_parts.append(f"  Max: {df[col].max()}")
                profile_parts.append(f"  Mean: {df[col].mean():.4f}")
                profile_parts.append(f"  Std: {df[col].std():.4f}")
            elif pd.api.types.is_string_dtype(dtype) or dtype == "object":
                if unique <= 10:
                    value_counts = df[col].value_counts().head(5)
                    profile_parts.append(f"  Top values: {dict(value_counts)}")

        return ToolResult(
            tool_call_id="",
            content="\n".join(profile_parts),
        )

    async def _sample(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Random sample of rows."""
        df = self._get_dataframe(source)
        n = options.get("n", 5)

        sample_df = df.sample(min(n, len(df)))

        return ToolResult(
            tool_call_id="",
            content=f"Random sample ({n} rows) from {source}:\n{sample_df.to_string()}",
        )

    async def _columns(self, source: str, options: dict[str, Any]) -> ToolResult:
        """List all columns with types."""
        df = self._get_dataframe(source)

        col_info = []
        for i, (col, dtype) in enumerate(df.dtypes.items()):
            col_info.append(f"{i:3d}. {col:<40} {str(dtype)}")

        return ToolResult(
            tool_call_id="",
            content=f"Columns in {source}:\n{'=' * 60}\n" + "\n".join(col_info),
        )

    async def _missing(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Analyze missing values."""
        df = self._get_dataframe(source)

        missing = df.isna().sum()
        missing_pct = (missing / len(df)) * 100

        # Only show columns with missing values
        has_missing = missing[missing > 0].sort_values(ascending=False)

        if len(has_missing) == 0:
            return ToolResult(
                tool_call_id="",
                content=f"No missing values in {source}",
            )

        lines = [f"Missing Values in {source}:", "=" * 60, ""]
        lines.append(f"{'Column':<40} {'Count':>10} {'Percent':>10}")
        lines.append("-" * 60)

        for col in has_missing.index:
            count = missing[col]
            pct = missing_pct[col]
            lines.append(f"{col:<40} {count:>10,} {pct:>9.1f}%")

        lines.append("-" * 60)
        lines.append(f"Total rows: {len(df):,}")
        lines.append(f"Columns with missing: {len(has_missing)} / {len(df.columns)}")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )

    async def _unique(self, source: str, options: dict[str, Any]) -> ToolResult:
        """Show unique value counts."""
        df = self._get_dataframe(source)

        columns = options.get("columns")
        if columns:
            if isinstance(columns, str):
                columns = [c.strip() for c in columns.split(",")]
        else:
            # Default to first 5 categorical-like columns
            columns = []
            for col in df.columns:
                if df[col].nunique() <= 20 or df[col].dtype == "object":
                    columns.append(col)
                    if len(columns) >= 5:
                        break

        lines = [f"Unique Values in {source}:", "=" * 60]

        for col in columns:
            if col not in df.columns:
                continue

            lines.append(f"\n{col}:")
            value_counts = df[col].value_counts().head(10)
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                lines.append(f"  {val}: {count:,} ({pct:.1f}%)")

            if df[col].nunique() > 10:
                lines.append(f"  ... and {df[col].nunique() - 10} more unique values")

        return ToolResult(
            tool_call_id="",
            content="\n".join(lines),
        )
