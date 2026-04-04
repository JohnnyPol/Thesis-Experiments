# Experiment 1: Single-Model Inference

This repository contains the code for Experiment 1 of the thesis project: evaluating a single ResNet-18 model under different inference setups.

This README focuses on the single-node part of Experiment 1:

- `exp1_1`: baseline ResNet-18 on one node
- `exp1_2`: early-exit ResNet-18 on one node

The distributed runs (`exp1_3`, `exp1_4`, `exp1_5`, `exp1_6`) build on the same model and config structure, but they are not the main focus of this document.

## What This Experiment Measures

The single-node pipeline records:

- inference latency
- throughput
- accuracy
- node utilization
- network byte deltas for the configured interface
- carbon emissions (`carbon_kg`)
- energy consumption (`energy_kWh`)
- exit usage statistics for the early-exit model

Outputs are saved as JSON and CSV files under `results/exp1_single_model/...`.

## Repository Layout

The files most relevant to Experiment 1 are:

- [`src/inference/single_node.py`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\src\inference\single_node.py): single-node inference entrypoint
- [`configs/experiments/exp1_1_baseline_single_node.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\experiments\exp1_1_baseline_single_node.yaml): baseline experiment config
- [`configs/experiments/exp1_2_ee_single_node.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\experiments\exp1_2_ee_single_node.yaml): early-exit experiment config
- [`configs/models/resnet18_baseline.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\models\resnet18_baseline.yaml): baseline model config
- [`configs/models/resnet18_ee_entropy.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\models\resnet18_ee_entropy.yaml): early-exit model config
- [`configs/datasets/cifar10.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\datasets\cifar10.yaml): dataset config
- [`configs/systems/single_node_worker1.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\systems\single_node_worker1.yaml): single-node system config
- [`scripts/run/run_exp1_1_baseline_single_node.sh`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\scripts\run\run_exp1_1_baseline_single_node.sh): baseline run script
- [`scripts/run/run_exp1_2_ee_single_node.sh`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\scripts\run\run_exp1_2_ee_single_node.sh): early-exit run script

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch and TorchVision separately if needed for your machine:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://www.piwheels.org/simple \
  torch
```

```bash
pip install torchvision --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://www.piwheels.org/simple
```

Set the project root on the Python path before running scripts manually:

```bash
export PYTHONPATH="$(pwd)"
```

## Data And Weights

The current single-node configs use:

- dataset: CIFAR-10
- model family: ResNet-18
- batch size: `1`
- device: `cpu`

Expected checkpoint paths:

- baseline: `checkpoints/resnet18_baseline.pth`
- early-exit: `checkpoints/resnet18_ee_entropy.pth`

The dataset config has `download: true`, so CIFAR-10 can be downloaded automatically into `./data` if it is not already present.

## Running Experiment 1

### 1. Baseline Single-Node Run

```bash
bash scripts/run/run_exp1_1_baseline_single_node.sh
```

This runs:

```bash
python -m src.inference.single_node \
  --config configs/experiments/exp1_1_baseline_single_node.yaml
```

Results are written to:

```text
results/exp1_single_model/01_single_node_baseline/run_001/
```

### 2. Early-Exit Single-Node Run

```bash
bash scripts/run/run_exp1_2_ee_single_node.sh
```

This runs:

```bash
python -m src.inference.single_node \
  --config configs/experiments/exp1_2_ee_single_node.yaml
```

Results are written to:

```text
results/exp1_single_model/02_single_node_ee/run_001/
```

## Output Files

Each single-node run saves:

- `metrics.json`: aggregate experiment metrics
- `latencies.csv`: per-sample latency and prediction records
- `resolved_config.json`: the fully resolved experiment, dataset, model, and system configuration

Important metrics in `metrics.json` include:

- `accuracy`
- `num_correct`
- `num_samples`
- `total_inference_time_sec`
- `throughput_samples_per_sec`
- `node_utilization`
- `carbon_kg`
- `energy_kWh`
- `network_rx_bytes`
- `network_tx_bytes`
- `network_total_bytes`

For the early-exit model, `metrics.json` also includes:

- `exit_0_count`, `exit_1_count`, `exit_2_count`, `exit_3_count`
- `exit_0_ratio`, `exit_1_ratio`, `exit_2_ratio`, `exit_3_ratio`

## Configuration Notes

Experiment configs follow a small composition pattern:

- the experiment YAML defines runtime and output settings
- `config_refs` point to dataset, model, and system YAML files
- the runtime loader resolves those into one bundle before execution

For example:

- [`exp1_1_baseline_single_node.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\experiments\exp1_1_baseline_single_node.yaml) uses the baseline model config
- [`exp1_2_ee_single_node.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\experiments\exp1_2_ee_single_node.yaml) uses the entropy-based early-exit model config

If you want to change experiment behavior, the most common places are:

- dataset root or loader settings in [`cifar10.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\datasets\cifar10.yaml)
- checkpoint path in the model config
- early-exit threshold in [`resnet18_ee_entropy.yaml`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\configs\models\resnet18_ee_entropy.yaml)
- output directory and warmup count in the experiment YAML

## Current Limitation

At the moment, the single-node entrypoint calls `evaluate_single_node(..., max_samples=1)` in [`single_node.py:384`](c:\Users\User\Desktop\Github_Projects\Thesis-Experiments\src\inference\single_node.py#L384). That means each run currently processes only one measured sample, even though the surrounding config and metric structure look like a full evaluation pipeline.

So if you run `exp1_1` or `exp1_2` right now, the produced metrics reflect a one-sample run after warmup, not the full CIFAR-10 test set.

If you want full-dataset evaluation, this line should be changed so `max_samples` comes from config or defaults to `None`.

## Reproducibility Tips

- Use the same checkpoint files for all comparisons.
- Keep `batch_size=1` when evaluating the early-exit model.
- Run baseline and early-exit experiments on the same machine under similar system load.
- Keep the configured network interface stable if you are comparing network byte deltas.
- Record package versions, especially `torch`, `torchvision`, `numpy`, and `codecarbon`.

## Summary

Experiment 1 compares a standard ResNet-18 against an early-exit ResNet-18 under the same single-node setup. The code is organized around small YAML configs and a single inference entrypoint, making it easy to rerun the baseline, rerun the early-exit model, or adjust thresholds, checkpoints, and output locations for thesis experiments.
