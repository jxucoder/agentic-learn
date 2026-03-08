# Experiments

This directory is for experiment setup generation, contestant configs, and
generated manifests. The reusable runtime stays under `src/aglearn/`; the
experiment CLI entrypoints live here and call into `aglearn_experiments`.

## Generate a setup

```bash
uv run python experiments/generate_setup.py \
  --task-type multiclass \
  --seed 42
```

This creates an experiment setup under `experiments/generated/<slug>/` with:

- `challenge.md` public Kaggle-style brief
- `manifest.json` machine-readable benchmark manifest
- `data/` containing train/test/sample/solution files

## Run the arena

```bash
uv run python experiments/run_arena.py \
  --manifest experiments/generated/multiclass-seed-42/manifest.json \
  --contestants experiments/configs/contestants.example.json
```

Arena results are written to `output/arena/<slug>/`.
