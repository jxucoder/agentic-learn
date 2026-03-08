"""Benchmark generation helpers for experiment workflows."""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import textwrap
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
    experiment_name: str
    theme: str | None
    benchmark_id: str
    slug: str
    title: str
    task_type: str
    metric: str
    target_column: str
    train_path: str
    validation_path: str
    validation_sample_submission_path: str
    test_path: str
    sample_submission_path: str
    solution_path: str
    meta_path: str
    validator_script_path: str
    evaluation_script_path: str
    submission_filename: str
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
    validation_path: str
    validation_sample_submission_path: str
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
    experiment_name: str | None = None,
    theme: str | None = None,
    allow_fallback: bool = False,
    brief_generator: BriefGenerator | None = None,
) -> BenchmarkManifest:
    """Generate a benchmark bundle and a Kaggle-style challenge brief."""
    output_root_path = Path(output_root).resolve()
    chosen_name = _choose_experiment_name(
        output_root=output_root_path,
        seed=seed,
        preferred_name=experiment_name,
    )
    slug = chosen_name
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
    prompt = _build_brief_prompt(bundle=bundle, meta=meta, theme=theme)

    generator = brief_generator or _generate_brief_with_gemini
    try:
        brief, brief_source = generator(prompt, gemini_model, benchmark_dir)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(f"Gemini brief generation failed: {exc}") from exc
        brief = _fallback_brief(meta=meta, task_type=task_type, theme=theme)
        brief_source = "template-fallback"

    normalized = _normalize_brief(brief, task_type=task_type, metric=bundle.metric)
    _ensure_theme_alignment(normalized, theme)
    validator_path = _write_submission_validator(benchmark_dir, bundle)
    evaluation_path = _write_validation_evaluator(benchmark_dir, bundle)
    public_description = _render_public_description(
        normalized,
        bundle=bundle,
        validator_script_path=validator_path,
        evaluation_script_path=evaluation_path,
    )
    agent_instructions = _build_agent_instructions(bundle)

    challenge_path = benchmark_dir / "challenge.md"
    manifest_path = benchmark_dir / "manifest.json"

    challenge_path.write_text(public_description, encoding="utf-8")
    manifest = BenchmarkManifest(
        experiment_name=chosen_name,
        theme=theme,
        benchmark_id=slug,
        slug=slug,
        title=normalized["title"],
        task_type=bundle.task_type,
        metric=bundle.metric,
        target_column=bundle.target_column,
        train_path=bundle.train_path,
        validation_path=bundle.validation_path,
        validation_sample_submission_path=bundle.validation_sample_submission_path,
        test_path=bundle.test_path,
        sample_submission_path=bundle.sample_submission_path,
        solution_path=bundle.solution_path,
        meta_path=bundle.meta_path,
        validator_script_path=str(validator_path),
        evaluation_script_path=str(evaluation_path),
        submission_filename="submission.csv",
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
        generate(task, output_dir=str(output_dir))
        meta_path = output_dir / f"synth_{task_name}_meta.json"
        meta = _load_json(meta_path)
        return _kaggle_bundle_from_full_dataset(
            full_path=Path(meta["files"]["full"]),
            meta_path=meta_path,
            task_type=task_type,
            metric="f1",
            target_column="target",
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
        generate(task, output_dir=str(output_dir))
        meta_path = output_dir / f"synth_{task_name}_meta.json"
        meta = _load_json(meta_path)
        return _kaggle_bundle_from_full_dataset(
            full_path=Path(meta["files"]["full"]),
            meta_path=meta_path,
            task_type=task_type,
            metric="r2",
            target_column="target",
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
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
) -> _Bundle:
    df = pd.read_csv(full_path)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))
        df.to_csv(full_path, index=False)

    n_train = int(len(df) * train_frac)
    n_validation = int(len(df) * validation_frac)
    n_test = len(df) - n_train - n_validation
    if n_train <= 0 or n_validation <= 0 or n_test <= 0:
        raise ValueError("Invalid train/validation/test split for benchmark bundle")

    train_df = df.iloc[:n_train].copy()
    validation_df = df.iloc[n_train : n_train + n_validation].copy()
    test_df = df.iloc[n_train + n_validation :].copy()
    test_features = test_df.drop(columns=[target_column]).copy()
    validation_submission = pd.DataFrame(
        {
            "row_id": validation_df["row_id"].astype(int),
            target_column: _default_submission_target(
                train_df, target_column, task_type
            ),
        }
    )
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
    validation_path = base_path.with_name(f"{base_path.name}_validation.csv")
    validation_sample_path = base_path.with_name(
        f"{base_path.name}_validation_sample_submission.csv"
    )
    test_path = base_path.with_name(f"{base_path.name}_test.csv")
    solution_path = base_path.with_name(f"{base_path.name}_solution.csv")
    sample_path = base_path.with_name(f"{base_path.name}_sample_submission.csv")

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    validation_submission.to_csv(validation_sample_path, index=False)
    test_features.to_csv(test_path, index=False)
    solution_df.to_csv(solution_path, index=False)
    sample_submission.to_csv(sample_path, index=False)

    meta = _load_json(meta_path)
    meta["kaggle_style"] = True
    meta["target_column"] = target_column
    meta["files"] = {
        "full": str(full_path.resolve()),
        "train": str(train_path.resolve()),
        "validation": str(validation_path.resolve()),
        "validation_sample_submission": str(validation_sample_path.resolve()),
        "test": str(test_path.resolve()),
        "solution": str(solution_path.resolve()),
        "sample_submission": str(sample_path.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return _Bundle(
        train_path=str(train_path.resolve()),
        validation_path=str(validation_path.resolve()),
        validation_sample_submission_path=str(validation_sample_path.resolve()),
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


def _build_brief_prompt(
    *,
    bundle: _Bundle,
    meta: dict[str, Any],
    theme: str | None,
) -> str:
    train_sample = pd.read_csv(bundle.train_path, nrows=5)
    schema_lines = []
    for column in train_sample.columns:
        dtype = str(train_sample[column].dtype)
        sample_value = train_sample.iloc[0][column]
        schema_lines.append(f"- {column}: dtype={dtype}, example={sample_value!r}")

    notes = meta.get("notes", [])
    hidden_notes = "\n".join(f"- {note}" for note in notes) or "- None provided"

    theme_block = ""
    if theme:
        theme_block = (
            f"Theme to anchor the public competition setup: {theme}\n"
            "Use the theme in the title, scenario, and dataset story while keeping the"
            " underlying task grounded in the actual schema.\n"
        )

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
        f"Validation CSV: {bundle.validation_path}\n"
        f"Test CSV: {bundle.test_path}\n"
        f"Validation sample submission CSV: {bundle.validation_sample_submission_path}\n"
        f"Sample submission CSV: {bundle.sample_submission_path}\n"
        f"{theme_block}"
        "Schema sample:\n"
        f"{chr(10).join(schema_lines)}\n\n"
        "Generation notes:\n"
        f"{hidden_notes}\n\n"
        "Constraints:\n"
        "- Make it feel like a serious Kaggle competition prompt.\n"
        "- Emphasize leaderboard robustness, validation discipline, and feature engineering.\n"
        "- Keep it specific to tabular ML, not generic AI benchmark language.\n"
        "- Do not use generic titles like 'Regression Benchmark' or 'Classification Benchmark'.\n"
        "- If a theme is provided, the title, short description, or scenario must clearly mention it.\n"
        "- Make the submission requirements concrete: a CSV with columns `row_id` and the target column.\n"
        "- Mention that competitors should use the sample submission to verify schema.\n"
        "- data_highlights, modeling_challenges, and submission_requirements must each be lists of 3-6 short bullets.\n"
    )


def _generate_brief_with_gemini(
    prompt: str,
    model: str | None,
    cwd: Path,
) -> tuple[dict[str, Any], str]:
    attempts: list[str] = []
    for candidate in _gemini_model_candidates(model):
        cmd = ["gemini", "--yolo", "--output-format", "json"]
        if candidate:
            cmd.extend(["--model", candidate])
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
            attempts.append(
                f"{candidate or 'cli-default'}: {detail or 'gemini CLI failed'}"
            )
            continue

        payload = _parse_brief_payload(proc.stdout)
        if not isinstance(payload, dict):
            attempts.append(
                f"{candidate or 'cli-default'}: Gemini response was not a JSON object"
            )
            continue
        return payload, f"gemini-cli:{candidate or 'cli-default'}"

    raise RuntimeError("; ".join(attempts) or "gemini CLI failed")


def _gemini_model_candidates(requested_model: str | None) -> list[str | None]:
    if requested_model:
        return [requested_model]

    installed = _detect_installed_gemini_models()
    ordered = [
        model_name
        for model_name in (
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        )
        if model_name in installed
    ]
    return ordered or [None]


def _detect_installed_gemini_models() -> set[str]:
    package_root = _gemini_cli_package_root()
    if package_root is None:
        return set()

    model_files = (
        package_root
        / "node_modules"
        / "@google"
        / "gemini-cli-core"
        / "dist"
        / "src"
        / "config"
        / "models.js",
        package_root
        / "node_modules"
        / "@google"
        / "gemini-cli-core"
        / "dist"
        / "src"
        / "config"
        / "defaultModelConfigs.js",
    )
    detected: set[str] = set()
    for model_file in model_files:
        if not model_file.exists():
            continue
        try:
            contents = model_file.read_text(encoding="utf-8")
        except OSError:
            continue
        detected.update(re.findall(r"gemini-[0-9][a-z0-9.\-]*", contents))
    return detected


def _gemini_cli_package_root() -> Path | None:
    gemini_path = shutil.which("gemini")
    if not gemini_path:
        return None

    resolved = Path(gemini_path).resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "gemini-cli" and parent.parent.name == "@google":
            return parent
    return None


def _parse_brief_payload(stdout: str) -> dict[str, Any]:
    candidate = stdout.strip()
    if not candidate:
        raise RuntimeError("Empty Gemini response")

    parsed = _extract_json_payload(candidate)
    if isinstance(parsed, dict):
        nested = _unwrap_brief_payload(parsed)
        if isinstance(nested, dict):
            return nested

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
            parsed = _extract_json_payload(value)
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


def _extract_json_payload(text: str) -> dict[str, Any] | list[Any] | None:
    parsed = _parse_json(text)
    if parsed is not None:
        return parsed

    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for block in fenced:
        parsed = _parse_json(block)
        if parsed is not None:
            return parsed

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        return _parse_json(brace_match.group(1))
    return None


def _fallback_brief(
    *,
    meta: dict[str, Any],
    task_type: str,
    theme: str | None,
) -> dict[str, Any]:
    theme_title = theme.title() if theme else task_type.replace("_", " ").title()
    theme_short = theme or "hidden signals"
    return {
        "title": f"{theme_title}: Hidden-Test Challenge",
        "short_description": (
            f"Build a robust tabular model for the {theme_short} setup that"
            " holds up on a hidden test split."
        ),
        "scenario": (
            f"You are given a labeled training table and an unlabeled competition"
            f" test table for the {theme_short} prediction task."
        ),
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
            "Use the public validation split to estimate model quality",
            "Generate predictions for every row in the provided test split",
            "Match the sample submission schema exactly",
        ],
        "evaluation_summary": (
            f"Submissions are ranked by {meta.get('metric', 'the configured metric')}"
            " on a hidden holdout."
        ),
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


def _ensure_theme_alignment(brief: dict[str, Any], theme: str | None) -> None:
    if not theme:
        return

    theme_tokens = [
        token for token in re.findall(r"[a-z0-9]+", theme.lower()) if len(token) >= 4
    ]
    if not theme_tokens:
        return

    combined_text = " ".join(
        [
            str(brief.get("title", "")),
            str(brief.get("short_description", "")),
            str(brief.get("scenario", "")),
        ]
    ).lower()
    if any(token in combined_text for token in theme_tokens):
        return

    raise RuntimeError(f"Generated brief did not reflect the requested theme: {theme}")


def _render_public_description(
    brief: dict[str, Any],
    *,
    bundle: _Bundle,
    validator_script_path: Path,
    evaluation_script_path: Path,
) -> str:
    target_column = bundle.target_column
    validation_name = Path(bundle.validation_path).name
    validation_sample_name = Path(bundle.validation_sample_submission_path).name
    test_name = Path(bundle.test_path).name
    sample_name = Path(bundle.sample_submission_path).name
    validator_name = validator_script_path.name
    evaluation_name = evaluation_script_path.name
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
        + "\n\n## Submission Format\n"
        + "- Final file name: `submission.csv`\n"
        + f"- Required columns: `row_id`, `{target_column}`\n"
        + f"- Include every row from `{test_name}` exactly once\n"
        + f"- Match the column order shown in `{sample_name}`\n"
        + "- Row order is optional, but duplicate or missing `row_id` values are invalid\n"
        + "\n\n## Public Validation\n"
        + f"- Fixed labeled validation split: `{validation_name}`\n"
        + f"- Validation prediction template: `{validation_sample_name}`\n"
        + "- Use the validation split for model selection and local score tracking\n"
        + "To score validation predictions, run:\n\n"
        + f"```bash\nuv run python {evaluation_name} --submission validation_submission.csv\n```\n\n"
        + "\n\n## Local Validation\n"
        + "From the experiment root, run:\n\n"
        + f"```bash\nuv run python {validator_name} --submission submission.csv\n```\n\n"
        + "The validator checks schema, row coverage, duplicate `row_id` values, and basic target parsing before leaderboard evaluation.\n"
        + f"\n\n## Evaluation\n{brief['evaluation_summary']}\n"
    )


def _build_agent_instructions(bundle: _Bundle) -> str:
    return (
        "This run is evaluated like a Kaggle competition.\n"
        "- Use the labeled train split at `data_path` for modeling and validation.\n"
        "- Use the `validation_data` resource as a fixed public holdout and `validation_sample_submission` for its prediction schema.\n"
        "- Use the `validation_evaluator` resource to score `validation_submission.csv` during development.\n"
        "- Use the `test_data` resource for final predictions, `sample_submission` for schema, and `submission_validator` for preflight validation.\n"
        "- Write `validation_submission.csv` for the validation rows and score it with the exact `validation_evaluator` path provided in the task resources.\n"
        f"- Write `submission.csv` in the current working directory with columns `row_id,{bundle.target_column}`.\n"
        "- Ensure `submission.csv` contains every test row exactly once and no duplicate `row_id` values.\n"
        "- Before finishing, run the exact `submission_validator` resource path against `submission.csv` and fix any reported errors.\n"
        "- The trusted runner uses `validation_submission.csv` plus the public evaluator for step selection, then scores the saved best `submission.csv` on the hidden test only once at the end.\n"
        f"- Optimize for hidden-test {bundle.metric}, not only local CV.\n"
    )


def _write_submission_validator(benchmark_dir: Path, bundle: _Bundle) -> Path:
    script_path = benchmark_dir / "validate_submission.py"
    script_path.write_text(
        _submission_validator_source(bundle=bundle),
        encoding="utf-8",
    )
    return script_path


def _submission_validator_source(bundle: _Bundle) -> str:
    test_relative_path = Path("data") / Path(bundle.test_path).name
    sample_relative_path = Path("data") / Path(bundle.sample_submission_path).name
    numeric_target = bundle.task_type in {"regression", "temporal_regression"}
    return textwrap.dedent(
        f"""\
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        import pandas as pd

        TARGET_COLUMN = {json.dumps(bundle.target_column)}
        EXPECTED_TEST_PATH = Path({json.dumps(str(test_relative_path))})
        SAMPLE_SUBMISSION_PATH = Path({json.dumps(str(sample_relative_path))})
        REQUIRE_NUMERIC_TARGET = {numeric_target}


        def validate_submission(submission_path: Path) -> list[str]:
            root = Path(__file__).resolve().parent
            expected_test_path = root / EXPECTED_TEST_PATH
            sample_submission_path = root / SAMPLE_SUBMISSION_PATH

            try:
                submission_df = pd.read_csv(submission_path)
            except Exception as exc:
                return [f"Unable to read submission CSV: {{exc}}"]

            try:
                expected_test_df = pd.read_csv(expected_test_path)
                sample_submission_df = pd.read_csv(sample_submission_path)
            except Exception as exc:
                return [f"Unable to read benchmark reference files: {{exc}}"]

            errors: list[str] = []
            expected_columns = list(sample_submission_df.columns)
            if list(submission_df.columns) != expected_columns:
                errors.append(
                    "Columns must exactly match the sample submission: "
                    + ", ".join(expected_columns)
                )
                return errors

            if submission_df["row_id"].isna().any():
                errors.append("`row_id` contains missing values.")
            if submission_df["row_id"].duplicated().any():
                errors.append("`row_id` contains duplicates.")
            if submission_df[TARGET_COLUMN].isna().any():
                errors.append(f"`{{TARGET_COLUMN}}` contains missing values.")

            expected_row_ids = set(expected_test_df["row_id"].tolist())
            actual_row_ids = set(submission_df["row_id"].tolist())
            missing_row_ids = sorted(expected_row_ids - actual_row_ids)[:10]
            extra_row_ids = sorted(actual_row_ids - expected_row_ids)[:10]
            if len(submission_df) != len(expected_test_df):
                errors.append(
                    f"Expected {{len(expected_test_df)}} rows but found {{len(submission_df)}}."
                )
            if missing_row_ids:
                errors.append(f"Missing row_id values (first 10): {{missing_row_ids}}")
            if extra_row_ids:
                errors.append(f"Unexpected row_id values (first 10): {{extra_row_ids}}")

            if REQUIRE_NUMERIC_TARGET:
                numeric_target = pd.to_numeric(submission_df[TARGET_COLUMN], errors="coerce")
                if numeric_target.isna().any():
                    errors.append(f"`{{TARGET_COLUMN}}` must be numeric for this task.")

            return errors


        def main() -> None:
            parser = argparse.ArgumentParser(
                description="Validate a competition submission against the public schema."
            )
            parser.add_argument("--submission", required=True)
            args = parser.parse_args()

            submission_path = Path(args.submission).resolve()
            errors = validate_submission(submission_path)
            payload = {{
                "ok": not errors,
                "submission_path": str(submission_path),
                "errors": errors,
            }}
            print(json.dumps(payload, indent=2))
            raise SystemExit(0 if not errors else 1)


        if __name__ == "__main__":
            main()
        """
    )


def _write_validation_evaluator(benchmark_dir: Path, bundle: _Bundle) -> Path:
    script_path = benchmark_dir / "evaluate_validation.py"
    script_path.write_text(
        _validation_evaluator_source(bundle=bundle),
        encoding="utf-8",
    )
    return script_path


def _validation_evaluator_source(bundle: _Bundle) -> str:
    validation_relative_path = Path("data") / Path(bundle.validation_path).name
    validation_sample_relative_path = (
        Path("data") / Path(bundle.validation_sample_submission_path).name
    )
    numeric_target = bundle.task_type in {"regression", "temporal_regression"}
    return textwrap.dedent(
        f"""\
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, r2_score

        METRIC = {json.dumps(bundle.metric)}
        TARGET_COLUMN = {json.dumps(bundle.target_column)}
        VALIDATION_PATH = Path({json.dumps(str(validation_relative_path))})
        VALIDATION_SAMPLE_SUBMISSION_PATH = Path(
            {json.dumps(str(validation_sample_relative_path))}
        )
        REQUIRE_NUMERIC_TARGET = {numeric_target}


        def validate_submission(submission_path: Path) -> list[str]:
            root = Path(__file__).resolve().parent
            validation_df = pd.read_csv(root / VALIDATION_PATH)
            sample_submission_df = pd.read_csv(root / VALIDATION_SAMPLE_SUBMISSION_PATH)

            try:
                submission_df = pd.read_csv(submission_path)
            except Exception as exc:
                return [f"Unable to read submission CSV: {{exc}}"]

            errors: list[str] = []
            expected_columns = list(sample_submission_df.columns)
            if list(submission_df.columns) != expected_columns:
                errors.append(
                    "Columns must exactly match the validation sample submission: "
                    + ", ".join(expected_columns)
                )
                return errors

            if submission_df["row_id"].isna().any():
                errors.append("`row_id` contains missing values.")
            if submission_df["row_id"].duplicated().any():
                errors.append("`row_id` contains duplicates.")
            if submission_df[TARGET_COLUMN].isna().any():
                errors.append(f"`{{TARGET_COLUMN}}` contains missing values.")

            expected_row_ids = set(validation_df["row_id"].tolist())
            actual_row_ids = set(submission_df["row_id"].tolist())
            missing_row_ids = sorted(expected_row_ids - actual_row_ids)[:10]
            extra_row_ids = sorted(actual_row_ids - expected_row_ids)[:10]
            if len(submission_df) != len(validation_df):
                errors.append(
                    f"Expected {{len(validation_df)}} rows but found {{len(submission_df)}}."
                )
            if missing_row_ids:
                errors.append(f"Missing row_id values (first 10): {{missing_row_ids}}")
            if extra_row_ids:
                errors.append(f"Unexpected row_id values (first 10): {{extra_row_ids}}")

            if REQUIRE_NUMERIC_TARGET:
                numeric_target = pd.to_numeric(submission_df[TARGET_COLUMN], errors="coerce")
                if numeric_target.isna().any():
                    errors.append(f"`{{TARGET_COLUMN}}` must be numeric for this task.")

            return errors


        def score_submission(submission_path: Path) -> tuple[float | None, list[str]]:
            errors = validate_submission(submission_path)
            if errors:
                return None, errors

            root = Path(__file__).resolve().parent
            validation_df = pd.read_csv(root / VALIDATION_PATH)
            submission_df = pd.read_csv(submission_path)
            merged = validation_df[["row_id", TARGET_COLUMN]].merge(
                submission_df[["row_id", TARGET_COLUMN]],
                on="row_id",
                how="left",
                suffixes=("_true", "_pred"),
                validate="one_to_one",
            )

            pred_column = f"{{TARGET_COLUMN}}_pred"
            true_column = f"{{TARGET_COLUMN}}_true"
            try:
                if METRIC == "f1":
                    score = float(
                        f1_score(
                            merged[true_column].astype(int),
                            merged[pred_column].astype(int),
                        )
                    )
                elif METRIC == "f1_macro":
                    score = float(
                        f1_score(
                            merged[true_column].astype(int),
                            merged[pred_column].astype(int),
                            average="macro",
                        )
                    )
                elif METRIC == "accuracy":
                    score = float(
                        accuracy_score(
                            merged[true_column].astype(int),
                            merged[pred_column].astype(int),
                        )
                    )
                elif METRIC == "r2":
                    score = float(
                        r2_score(
                            merged[true_column].astype(float),
                            merged[pred_column].astype(float),
                        )
                    )
                else:
                    return None, [f"Unsupported metric: {{METRIC}}"]
            except (TypeError, ValueError) as exc:
                return None, [f"Unable to score submission: {{exc}}"]

            return score, []


        def main() -> None:
            parser = argparse.ArgumentParser(
                description="Score a submission against the public validation split."
            )
            parser.add_argument("--submission", required=True)
            args = parser.parse_args()

            submission_path = Path(args.submission).resolve()
            score, errors = score_submission(submission_path)
            payload = {{
                "ok": not errors,
                "submission_path": str(submission_path),
                "metric_name": METRIC,
                "metric": score,
                "errors": errors,
            }}
            print(json.dumps(payload, indent=2))
            raise SystemExit(0 if not errors else 1)


        if __name__ == "__main__":
            main()
        """
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


_ADJECTIVES = (
    "amber",
    "arcane",
    "brisk",
    "candid",
    "cinder",
    "clear",
    "cobalt",
    "crisp",
    "ember",
    "frozen",
    "gentle",
    "hidden",
    "ivory",
    "lunar",
    "mellow",
    "nimble",
    "opal",
    "quiet",
    "rapid",
    "scarlet",
    "silent",
    "silver",
    "steady",
    "velvet",
)

_NOUNS = (
    "anchor",
    "asteroid",
    "beacon",
    "brook",
    "canyon",
    "circuit",
    "cloud",
    "comet",
    "delta",
    "falcon",
    "forest",
    "harbor",
    "matrix",
    "meadow",
    "nebula",
    "orbit",
    "otter",
    "pine",
    "radar",
    "reef",
    "signal",
    "summit",
    "thunder",
    "vector",
)


def _choose_experiment_name(
    *,
    output_root: Path,
    seed: int,
    preferred_name: str | None,
) -> str:
    if preferred_name:
        normalized = _normalize_experiment_name(preferred_name)
        if (output_root / normalized).exists():
            raise ValueError(f"experiment_name already exists: {normalized}")
        return normalized

    rng = random.Random(seed ^ random.SystemRandom().randrange(1 << 30))
    for _ in range(128):
        candidate = f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"
        if not (output_root / candidate).exists():
            return candidate
    raise RuntimeError("Unable to allocate a unique experiment name")


def _normalize_experiment_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise ValueError("experiment_name must contain letters or digits")
    return normalized
