# Experiment and Reproduction Guide

This guide documents the configurations corresponding to the final thesis and the operational constraints of the current code. Run commands from the repository root.

## Final evaluated matrix

The stored thesis results comprise 28 runs: 14 configurations on CIFAR-10 and the same 14 on CIFAR-10.1.

| Group | Models | Instances | Execution | CIFAR-10 tasks | CIFAR-10.1 tasks |
| --- | --- | ---: | --- | ---: | ---: |
| Exp1.1 | ResNet-18/34 baseline | 1 | Single node | 10,000 | 2,000 |
| Exp1.2 | ResNet-18/34 EE | 1 | Single node | 10,000 | 2,000 |
| Exp1.3 | ResNet-18/34 EE | 1 | Canonical three-worker route | 10,000 | 2,000 |
| Exp2 | ResNet-18/34 EE | 2 | Shared three-worker route | 20,000 | 4,000 |
| Exp3.1–3.3 | ResNet-18/34 EE | 2 | Three static placements | 20,000 | 4,000 |

CIFAR-10.1 uses the CIFAR-10 checkpoints without retraining, fine-tuning, or a threshold change.

## Configuration composition

Every experiment config references one dataset, model, and system config. The loader resolves them relative to the repository root and stores the complete bundle in `resolved_config.json`.

| Dataset | Suffix | Data behavior |
| --- | --- | --- |
| CIFAR-10 | `_cifar10` | TorchVision download enabled |
| CIFAR-10.1 v6 | `_cifar10_1` | Manual NumPy files under `data/cifar10.1/` |

Final configs use batch size 1, 20 warm-up samples, per-task metrics, and a fixed `run_001` output path.

## Partition definition

```text
partition 0: conv1 -> maxpool -> layer0 -> exit0 -> layer1 -> exit1
partition 1: layer2 -> exit2
partition 2: layer3 -> adaptive average pooling -> final classifier (exit3)
```

Partition 0 receives every task. Later partitions receive only tasks that do not exit earlier.

## Experiment catalog

| Group | Entrypoint | Description |
| --- | --- | --- |
| Exp1.1 | `src.inference.single_node` | Full-depth baseline on one node |
| Exp1.2 | `src.inference.single_node` | Early-exit model on one node |
| Exp1.3 | `src.distributed.master_client` | One early-exit model across three workers |
| Exp2/3 | `src.distributed.multi_model_master_client` | Two concurrent logical instances |

Experiment 1.3 route:

```text
model_0: worker1/partition0 -> worker2/partition1 -> worker3/partition2
```

Experiment 2 uses that route for both `model_0` and `model_1`.

Experiment 3 changes only explicit YAML assignments/routes:

| Placement | Model 0 route | Model 1 route |
| --- | --- | --- |
| Exp3.1 — first split, late consolidated | `worker1 → worker3 → worker3` | `worker2 → worker3 → worker3` |
| Exp3.2 — staggered late | `worker1 → worker2 → worker3` | `worker2 → worker3 → worker3` |
| Exp3.3 — cross-balanced | `worker1 → worker3 → worker2` | `worker2 → worker1 → worker3` |

Adjacent partitions assigned to the same worker execute locally; otherwise the worker serializes and forwards the activation over HTTP.

## Prerequisites

1. Complete the root [`README.md`](../README.md#installation) setup and Git LFS checkout on all nodes.
2. Use the same revision and checkpoints on the master and workers.
3. Put CIFAR-10.1 v6 data/label NumPy files in `data/cifar10.1/` on the master.
4. Confirm `connect_host` and ports in `configs/systems/homogeneous_3workers.yaml`.
5. Copy a config and change `output.dir` before rerunning if the published `run_001` must be preserved.

## Single-node execution

```bash
# CIFAR-10
bash scripts/run/run_exp1_1_baseline_single_node_cifar10.sh
bash scripts/run/run_exp1_2_ee_single_node_cifar10.sh

# CIFAR-10.1
bash scripts/run/run_exp1_1_baseline_single_node_cifar10_1.sh
bash scripts/run/run_exp1_2_ee_single_node_cifar10_1.sh
```

Individual config:

```bash
python -m src.inference.single_node \
  --config configs/experiments/exp1_2_resnet18_ee_single_node_cifar10.yaml
```

## Distributed execution

Start one API process on each worker using the exact same config:

```bash
CONFIG=configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10.yaml
bash scripts/run/start_worker_api.sh "$CONFIG" worker1  # matching device
bash scripts/run/start_worker_api.sh "$CONFIG" worker2
bash scripts/run/start_worker_api.sh "$CONFIG" worker3
```

The API exposes `/health`, `/info`, `/infer`, `/monitoring/start`, and `/monitoring/stop`. On the master:

```bash
bash scripts/run/healthcheck.sh "$CONFIG"
bash scripts/run/run_exp1_3_ee_3nodes_cifar10.sh "$CONFIG"
```

Use the matching master wrapper:

| Group | CIFAR-10 | CIFAR-10.1 |
| --- | --- | --- |
| Exp1.3 | `run_exp1_3_ee_3nodes_cifar10.sh` | `run_exp1_3_ee_3nodes_cifar10_1.sh` |
| Exp2 | `run_exp2_homogeneous_multi_model_cifar10.sh` | `run_exp2_homogeneous_multi_model_cifar10_1.sh` |
| Exp3 | `run_exp3_memory_aware_multi_model_cifar10.sh` | `run_exp3_memory_aware_multi_model_cifar10_1.sh` |

Examples after matching workers are running:

```bash
bash scripts/run/run_exp2_homogeneous_multi_model_cifar10.sh \
  configs/experiments/exp2_resnet34_cifar10.yaml

bash scripts/run/run_exp3_memory_aware_multi_model_cifar10_1.sh \
  configs/experiments/exp3_3_resnet34_cifar10_1.yaml
```

### Worker lifecycle constraint

Restart all workers whenever the config changes—including model depth, dataset, or Experiment 3 placement. The health check validates reachability, base worker/partition identity, model, and dataset; `/info` does not expose the full placement map, so same-model/same-dataset placements are not distinguishable by that check.

Distributed wrappers accept multiple configs, but a worker loads one config at startup. Run one explicit config per worker lifecycle.

## Outputs and thesis artifacts

Current entrypoints write `metrics.json`, `inference_times.csv`, and `resolved_config.json`. Archived CIFAR-10 runs use `latencies.csv` and legacy latency field names.

```bash
python -m src.visualization.tables --dataset-name cifar10 --table all --best-experiment 3.2
python -m src.visualization.plots --dataset-name cifar10 --plot all --best-experiment 3.2
```

For CIFAR-10.1, use `--dataset-name cifar10_1 --best-experiment 3.3`. Tables are emitted as CSV/LaTeX and plots as PNG/PDF below `results/thesis_visualizations/`.

The `generate_exp*_thesis_artifacts.{sh,ps1}` wrappers are incomplete because they invoke absent `src.visualization.summary`. Use the direct modules above.

## Implemented versus future work

Implemented and evaluated: CIFAR-10/CIFAR-10.1 inference, baseline and three-exit ResNet-18/34, fixed three-way partitioning, one/two-instance execution, three static placements, HTTP forwarding, and stored metrics/artifacts.

CIFAR-100 loader/config/script paths are scaffolding: their checkpoint files and stored results are absent, and CIFAR-100 is future evaluation in the thesis.

Not implemented in the evaluated system: online placement, threshold sweeps, joint split/placement optimization, heterogeneous workers, Grafana/Prometheus dashboards, compressed activation transport, other model families, and external power measurement.
