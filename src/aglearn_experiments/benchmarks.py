"""Benchmark generation helpers for experiment workflows."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from aglearn.data import (
    HardSyntheticTask,
    SyntheticTask,
    generate,
    generate_high_dim,
    generate_multiclass,
    generate_temporal_regression,
)


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_id: str
    slug: str
    title: str
    task_type: str
    metric: str
    target_column: str
    train_path: str
    test_path: str
    sample_submission_path: str
    solution_path: str
    meta_path: str
    challenge_markdown_path: str
    public_description: str
    agent_instructions: str
    seed: int
    brief_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Bundle:
    train_path: str
    test_path: str
    sample_submission_path: str
    solution_path: str
    meta_path: str
    task_type: str
    metric: str
    target_column: str
    seed: int


BriefGenerator = Callable[[str, str | None, Path], tuple[dict[str, Any], str]]


def generate_benchmark(
    *,
    task_type: str,
    seed: int,
    samples: int,
    noise: float,
    output_root: str = "experiments/generated",
    gemini_model: str | None = None,
    brief_generator: BriefGenerator | None = None,
) -> BenchmarkManifest:
    """Generate a benchmark bundle and a Kaggle-style challenge brief."""
    slug = f"{task_type.replace('_', '-')}-seed-{seed}"
    benchmark_dir = Path(output_root).resolve() / slug
    data_dir = benchmark_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    bundle = _materialize_bundle(
        task_type=task_type,
        seed=seed,
        samples=samples,
        noise=noise,
        output_dir=data_dir,
        slug=slug,
    )
    meta = _load_json(Path(bundle.meta_path))
    prompt = _build_brief_prompt(bundle=bundle, meta=meta)

    generator = brief_generator or _generate_brief_with_gemini
    try:
        brief, brief_source = generator(prompt, gemini_model, benchmark_dir)
    except Exception:
        brief = _fallback_brief(meta=meta, task_type=task_type)
        brief_source = "template-fallback"

    normalized = _normalize_brief(brief, task_type=task_type, metric=bundle.metric)
    public_description = _render_public_description(normalized)
    agent_instructions = _build_agent_instructions(bundle)

    challenge_path = benchmark_dir / "challenge.md"
    manifest_path = benchmark_dir / "manifest.json"

    challenge_path.write_text(public_description, encoding="utf-8")
    manifest = BenchmarkManifest(
        benchmark_id=slug,
        slug=slug,
        title=normalized["title"],
        task_type=bundle.task_type,
        metric=bundle.metric,
        target_column=bundle.target_column,
        train_path=bundle.train_path,
        test_path=bundle.test_path,
        sample_submission_path=bundle.sample_submission_path,
        solution_path=bundle.solution_path,
        meta_path=bundle.meta_path,
        challenge_markdown_path=str(challenge_path),
        public_description=public_description,
        agent_instructions=agent_instructions,
        seed=bundle.seed,
        brief_source=brief_source,
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


def _materialize_bundle(
    *,
    task_type: str,
    seed: int,
    samples: int,
    noise: float,
    output_dir: Path,
    slug: str,
) -> _Bundle:
    task_name = slug.replace("-", "_")
    if task_type == "classification":
        task = SyntheticTask(
            name=task_name,
            task_type="classification",
            metric="f1",
            n_samples=samples,
            noise_level=noise,
            seed=seed,
        )
        train_path = generate(task, output_dir=str(output_dir))
        meta_path = output_dir / f"synth_{task_name}_meta.json"
        meta = _load_json(meta_path)
        return _Bundle(
            train_path=train_path,
            test_path=meta["files"]["test"],
            sample_submission_path=meta["files"]["sample_submission"],
            solution_path=meta["files"]["solution"],
            meta_path=str(meta_path.resolve()),
            task_type=task_type,
            metric="f1",
            target_column="target",
            seed=seed,
        )

    if task_type == "regression":
        task = SyntheticTask(
            name=task_name,
            task_type="regression",
            metric="r2",
            n_samples=samples,
            noise_level=noise,
            seed=seed,
        )
        train_path = generate(task, output_dir=str(output_dir))
        meta_path = output_dir / f"synth_{task_name}_meta.json"
        meta = _load_json(meta_path)
        return _Bundle(
            train_path=train_path,
            test_path=meta["files"]["test"],
            sample_submission_path=meta["files"]["sample_submission"],
            solution_path=meta["files"]["solution"],
            meta_path=str(meta_path.resolve()),
            task_type=task_type,
            metric="r2",
            target_column="target",
            seed=seed,
        )

    hard_metric = {
        "multiclass": "f1_macro",
        "temporal_regression": "r2",
        "high_dim": "f1",
    }.get(task_type)
    if hard_metric is None:
        raise ValueError(f"Unsupported task_type: {task_type}")

    hard_task = HardSyntheticTask(
        name=task_name,
        task_type=task_type,
        metric=hard_metric,
        n_samples=samples,
        noise_level=noise,
        seed=seed,
    )
    generators = {
        "multiclass": generate_multiclass,
        "temporal_regression": generate_temporal_regression,
        "high_dim": generate_high_dim,
    }
    full_path = generators[task_type](hard_task, output_dir=str(output_dir))
    meta_path = output_dir / f"synth_{task_name}_meta.json"
    return _kaggle_bundle_from_full_dataset(
        full_path=Path(full_path),
        meta_path=meta_path,
        task_type=task_type,
        metric=hard_metric,
        target_column="target",
    )


def _kaggle_bundle_from_full_dataset(
    *,
    full_path: Path,
    meta_path: Path,
    task_type: str,
    metric: str,
    target_column: str,
    train_frac: float = 0.8,
) -> _Bundle:
    df = pd.read_csv(full_path)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))
        df.to_csv(full_path, index=False)

    n_train = int(len(df) * train_frac)
    if n_train <= 0 or n_train >= len(df):
        raise ValueError("Invalid train/test split for benchmark bundle")

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    test_features = test_df.drop(columns=[target_column]).copy()
    solution_df = test_df[["row_id", target_column]].copy()
    sample_submission = pd.DataFrame(
        {
            "row_id": test_features["row_id"].astype(int),
            target_column: _default_submission_target(
                train_df, target_column, task_type
            ),
        }
    )

    base_path = full_path.with_suffix("")
    train_path = base_path.with_name(f"{base_path.name}_train.csv")
    test_path = base_path.with_name(f"{base_path.name}_test.csv")
    solution_path = base_path.with_name(f"{base_path.name}_solution.csv")
    sample_path = base_path.with_name(f"{base_path.name}_sample_submission.csv")

    train_df.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    solution_df.to_csv(solution_path, index=False)
    sample_submission.to_csv(sample_path, index=False)

    meta = _load_json(meta_path)
    meta["kaggle_style"] = True
    meta["target_column"] = target_column
    meta["files"] = {
        "full": str(full_path.resolve()),
        "train": str(train_path.resolve()),
        "test": str(test_path.resolve()),
        "solution": str(solution_path.resolve()),
        "sample_submission": str(sample_path.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return _Bundle(
        train_path=str(train_path.resolve()),
        test_path=str(test_path.resolve()),
        sample_submission_path=str(sample_path.resolve()),
        solution_path=str(solution_path.resolve()),
        meta_path=str(meta_path.resolve()),
        task_type=task_type,
        metric=metric,
        target_column=target_column,
        seed=int(meta.get("seed", 0)),
    )


def _default_submission_target(
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> Any:
    if task_type in {"regression", "temporal_regression"}:
        return round(float(train_df[target_column].mean()), 6)
    return 0


def _build_brief_prompt(*, bundle: _Bundle, meta: dict[str, Any]) -> str:
    train_sample = pd.read_csv(bundle.train_path, nrows=5)
    schema_lines = []
    for column in train_sample.columns:
        dtype = str(train_sample[column].dtype)
        sample_value = train_sample.iloc[0][column]
        schema_lines.append(f"- {column}: dtype={dtype}, example={sample_value!r}")

    notes = meta.get("notes", [])
    hidden_notes = "\n".join(f"- {note}" for note in notes) or "- None provided"

    return (
        "You are writing a hard Kaggle-style tabular ML competition brief.\n"
        "Write only the public-facing challenge. Do not reveal hidden label-generation logic,\n"
        "the held-out solution file, or synthetic shortcuts.\n"
        "Return valid JSON with these keys:\n"
        "title, short_description, scenario, objective, data_highlights,\n"
        "modeling_challenges, submission_requirements, evaluation_summary.\n\n"
        f"Task type: {bundle.task_type}\n"
        f"Metric: {bundle.metric}\n"
        f"Target column: {bundle.target_column}\n"
        f"Train CSV: {bundle.train_path}\n"
        f"Test CSV: {bundle.test_path}\n"
        f"Sample submission CSV: {bundle.sample_submission_path}\n"
        "Schema sample:\n"
        + "\n".join(schema_lines)
        + "\n\nGeneration notes:\n"
        + hidden_notes
        + "\n\nConstraints:\n"
        "- Make it feel like a serious Kaggle competition prompt.\n"
        "- Emphasize leaderboard robustness, validation discipline, and feature engineering.\n"
        "- Keep it specific to tabular ML, not generic AI benchmark language.\n"
        "- data_highlights, modeling_challenges, and submission_requirements must each be lists of 3-6 short bullets.\n"
    )


def _generate_brief_with_gemini(
    prompt: str,
    model: str | None,
    cwd: Path,
) -> tuple[dict[str, Any], str]:
    cmd = ["gemini", "--yolo", "--output-format", "json"]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail or "gemini CLI failed")

    payload = _parse_brief_payload(proc.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("gemini CLI did not return a JSON object")
    return payload, "gemini-cli"


def _parse_brief_payload(stdout: str) -> dict[str, Any]:
    candidate = stdout.strip()
    if not candidate:
        raise RuntimeError("Empty Gemini response")

    parsed = _parse_json(candidate)
    if isinstance(parsed, dict):
        nested = _unwrap_brief_payload(parsed)
        if isinstance(nested, dict):
            return nested

    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", stdout, flags=re.DOTALL)
    for block in fenced:
        parsed = _parse_json(block)
        if isinstance(parsed, dict):
            return parsed

    brace_match = re.search(r"(\{.*\})", stdout, flags=re.DOTALL)
    if brace_match:
        parsed = _parse_json(brace_match.group(1))
        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError("Unable to parse Gemini brief payload")


def _unwrap_brief_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "title" in payload and "scenario" in payload:
        return payload
    for key in ("result", "output", "content", "text", "response"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = _unwrap_brief_payload(value)
            if nested:
                return nested
        if isinstance(value, str):
            parsed = _parse_json(value)
            if isinstance(parsed, dict):
                nested = _unwrap_brief_payload(parsed)
                if nested:
                    return nested
    return None


def _parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fallback_brief(*, meta: dict[str, Any], task_type: str) -> dict[str, Any]:
    return {
        "title": f"Hidden Signals: {task_type.replace('_', ' ').title()} Challenge",
        "short_description": "Build a robust tabular model that holds up on a hidden test split.",
        "scenario": "You are given a labeled training table and an unlabeled competition test table.",
        "objective": "Maximize the leaderboard metric without overfitting to spurious structure.",
        "data_highlights": meta.get("notes", [])[:3]
        or [
            "Mixed feature types with realistic noise patterns",
            "Leaderboard-style hidden evaluation split",
            "Synthetic but nontrivial tabular structure",
        ],
        "modeling_challenges": [
            "Distribution shift between train and test",
            "Distractor features and brittle shortcuts",
            "Validation strategy matters as much as model choice",
        ],
        "submission_requirements": [
            "Train on the labeled split only",
            "Generate predictions for every row in the provided test split",
            "Match the sample submission schema exactly",
        ],
        "evaluation_summary": f"Submissions are ranked by {meta.get('metric', 'the configured metric')} on a hidden holdout.",
    }


def _normalize_brief(
    brief: dict[str, Any],
    *,
    task_type: str,
    metric: str,
) -> dict[str, Any]:
    return {
        "title": str(brief.get("title") or f"{task_type.title()} Benchmark"),
        "short_description": str(
            brief.get("short_description") or "A hard Kaggle-style tabular benchmark."
        ),
        "scenario": str(
            brief.get("scenario")
            or "You are competing on a tabular prediction task with a hidden leaderboard."
        ),
        "objective": str(
            brief.get("objective") or f"Maximize {metric} on the hidden test set."
        ),
        "data_highlights": _coerce_lines(
            brief.get("data_highlights"),
            fallback=["Mixed tabular inputs", "Unlabeled test set", "Realistic noise"],
        ),
        "modeling_challenges": _coerce_lines(
            brief.get("modeling_challenges"),
            fallback=[
                "Robust validation is required",
                "Spurious signal can break generalization",
                "Feature engineering matters",
            ],
        ),
        "submission_requirements": _coerce_lines(
            brief.get("submission_requirements"),
            fallback=[
                "Use the provided sample submission schema",
                "Predict every test row exactly once",
                "Optimize for hidden-set performance",
            ],
        ),
        "evaluation_summary": str(
            brief.get("evaluation_summary")
            or f"Ranked by {metric} on a hidden evaluation split."
        ),
    }


def _coerce_lines(value: Any, *, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned
    if isinstance(value, str) and value.strip():
        return [line.strip("- ").strip() for line in value.splitlines() if line.strip()]
    return fallback


def _render_public_description(brief: dict[str, Any]) -> str:
    return (
        f"# {brief['title']}\n\n"
        f"{brief['short_description']}\n\n"
        "## Scenario\n"
        f"{brief['scenario']}\n\n"
        "## Objective\n"
        f"{brief['objective']}\n\n"
        "## Data Highlights\n"
        + "\n".join(f"- {line}" for line in brief["data_highlights"])
        + "\n\n## Modeling Challenges\n"
        + "\n".join(f"- {line}" for line in brief["modeling_challenges"])
        + "\n\n## Submission Requirements\n"
        + "\n".join(f"- {line}" for line in brief["submission_requirements"])
        + f"\n\n## Evaluation\n{brief['evaluation_summary']}\n"
    )


def _build_agent_instructions(bundle: _Bundle) -> str:
    return (
        "This run is evaluated like a Kaggle competition.\n"
        "- Use the labeled train split at `data_path` for modeling and validation.\n"
        "- Use the `test_data` resource for final predictions and `sample_submission` for the required schema.\n"
        "- Write `submission.csv` in the current working directory with columns `row_id,target`.\n"
        "- You may report your own CV metric in `result.json`, but leaderboard scoring is computed externally from `submission.csv`.\n"
        f"- Optimize for hidden-test {bundle.metric}, not only local CV.\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
