# Jetson Nano Legacy Runtime

The repository includes an isolated worker runtime for Jetson Nano devices running JetPack 4.x, L4T R32.x, and Python 3.6.

Use this runtime only on the Jetson Nano. The main application path remains unchanged.

## Why This Exists

The modern worker stack uses newer Python syntax and framework dependencies that are not a good fit for the Jetson Nano legacy software stack.

The legacy path:

- avoids `from __future__ import annotations`
- avoids `list[str]`, `dict[str, int]`, `X | Y`, and similar modern syntax
- avoids FastAPI and Pydantic on the Jetson path
- keeps the same external HTTP contract used by the master

## Entry Point

Run the legacy worker with:

```bash
bash scripts/run/start_jetson_legacy_worker.sh <experiment_config> jetson
```

Examples:

```bash
bash scripts/run/start_jetson_legacy_worker.sh configs/experiments/exp1_5_ee_heterogeneous.yaml jetson
```

```bash
bash scripts/run/start_jetson_legacy_worker.sh configs/experiments/exp1_6_ee_heterogeneous_3nodes.yaml jetson
```

## Notes

- This legacy runtime currently supports the Jetson only as the final worker in the pipeline.
- That matches the current experiment topology for `exp1_5` and `exp1_6`.
- Install Jetson-compatible `torch` and `torchvision` separately, then install the lightweight extras from `requirements-jetson-py36.txt`.
