# Thesis Experiments: Early-Exit ResNet Inference

This repository contains experiment code, configs, run scripts, metrics, and
thesis visualization tools for evaluating baseline ResNet and early-exit
ResNet inference across single-node, distributed single-model, and distributed
multi-model deployments.

## Environment Setup

Run the setup script:

```bash
scripts/setup_environment.sh
```
To force installation of CPU PyTorch and TorchVision:

```bash
scripts/setup_environment.sh --with-torch
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the repository dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch and TorchVision separately when needed for Raspberry Pi CPU
execution:

```bash
pip install torchvision --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://www.piwheels.org/simple
```

Set the project root on `PYTHONPATH` before running modules manually:

```bash
export PYTHONPATH="$(pwd)"
```

## Experiment Catalog

| ID | Config | Model | Topology | Entrypoint |
| --- | --- | --- | --- | --- |
| `exp1_1_resnet18_cifar10` | `configs/experiments/exp1_1_resnet18_baseline_single_node_cifar10.yaml` | ResNet-18 baseline | one CPU worker | `src.inference.single_node` |
| `exp1_1_resnet34_cifar10` | `configs/experiments/exp1_1_resnet34_baseline_single_node_cifar10.yaml` | ResNet-34 baseline | one CPU worker | `src.inference.single_node` |
| `exp1_2_resnet18_cifar10` | `configs/experiments/exp1_2_resnet18_ee_single_node_cifar10.yaml` | ResNet-18 EE | one CPU worker | `src.inference.single_node` |
| `exp1_2_resnet34_cifar10` | `configs/experiments/exp1_2_resnet34_ee_single_node_cifar10.yaml` | ResNet-34 EE | one CPU worker | `src.inference.single_node` |
| `exp1_3_resnet18_cifar10` | `configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10.yaml` | ResNet-18 EE | three CPU workers | `src.distributed.master_client` |
| `exp1_3_resnet34_cifar10` | `configs/experiments/exp1_3_resnet34_ee_3nodes_cifar10.yaml` | ResNet-34 EE | three CPU workers | `src.distributed.master_client` |
| `exp2_resnet18_cifar10` | `configs/experiments/exp2_resnet18_cifar10.yaml` | ResNet-18 EE / ResNet-18 EE | two logical models on three CPU workers | `src.distributed.multi_model_master_client` |
| `exp2_resnet34_cifar10` | `configs/experiments/exp2_resnet34_cifar10.yaml` | ResNet-34 EE / ResNet-34 EE | two logical models on three CPU workers | `src.distributed.multi_model_master_client` |
| `exp3_1_resnet18_cifar10` | `configs/experiments/exp3_1_resnet18_cifar10.yaml` | ResNet-18 EE / ResNet-18 EE | 3.1 consolidated-late placement | `src.distributed.multi_model_master_client` |
| `exp3_1_resnet34_cifar10` | `configs/experiments/exp3_1_resnet34_cifar10.yaml` | ResNet-34 EE / ResNet-34 EE | 3.1 consolidated-late placement | `src.distributed.multi_model_master_client` |
| `exp3_2_resnet18_cifar10` | `configs/experiments/exp3_2_resnet18_cifar10.yaml` | ResNet-18 EE / ResNet-18 EE | 3.2 staggered-late placement | `src.distributed.multi_model_master_client` |
| `exp3_2_resnet34_cifar10` | `configs/experiments/exp3_2_resnet34_cifar10.yaml` | ResNet-34 EE / ResNet-34 EE | 3.2 staggered-late placement | `src.distributed.multi_model_master_client` |
| `exp3_3_resnet18_cifar10` | `configs/experiments/exp3_3_resnet18_cifar10.yaml` | ResNet-18 EE / ResNet-18 EE | 3.3 cross-balanced placement | `src.distributed.multi_model_master_client` |
| `exp3_3_resnet34_cifar10` | `configs/experiments/exp3_3_resnet34_cifar10.yaml` | ResNet-34 EE / ResNet-34 EE | 3.3 cross-balanced placement | `src.distributed.multi_model_master_client` |

The three-worker partitioning used by distributed experiments is:

```text
worker1: conv1 -> maxpool -> layer0 -> exit0 -> layer1 -> exit1
worker2: layer2 -> exit2
worker3: layer3 -> avgpool -> fc
```

Experiment 3 changes only the model-partition placement declared in each
experiment YAML.

Experiment config filenames and experiment IDs explicitly include the dataset
suffix, for example `configs/experiments/exp2_resnet18_cifar10.yaml` and
`configs/experiments/exp2_resnet18_cifar100.yaml`.

## Data And Checkpoints

The implemented configs run CIFAR-10 and CIFAR-100 with `batch_size: 1`.
CIFAR-10 checkpoints are loaded from:

- `checkpoints/resnet18_baseline.pth`
- `checkpoints/resnet34_baseline.pth`
- `checkpoints/resnet18_ee_entropy.pth`
- `checkpoints/resnet34_ee_entropy.pth`

CIFAR-100 checkpoints are loaded from:

- `checkpoints/resnet18_baseline_cifar100.pth`
- `checkpoints/resnet34_baseline_cifar100.pth`
- `checkpoints/resnet18_ee_entropy_cifar100.pth`
- `checkpoints/resnet34_ee_entropy_cifar100.pth`

Configured checkpoint files must exist on every node before running the
corresponding experiment.

## Running

Single-node runs:

```bash
# Run baseline variants by dataset.
bash scripts/run/run_exp1_1_baseline_single_node_cifar10.sh
bash scripts/run/run_exp1_1_baseline_single_node_cifar100.sh

# Run single-node early-exit variants by dataset.
bash scripts/run/run_exp1_2_ee_single_node_cifar10.sh
bash scripts/run/run_exp1_2_ee_single_node_cifar100.sh
```

Distributed workers:

```bash
bash scripts/run/start_worker_api.sh configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10.yaml worker1
bash scripts/run/start_worker_api.sh configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10.yaml worker2
bash scripts/run/start_worker_api.sh configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10.yaml worker3
```

For distributed runs, each worker must be restarted with the same experiment
config that the master is about to run. The distributed run scripts check
`/info` before each run and stop if a worker is still serving a different
model/dataset.

Distributed masters:

```bash
bash scripts/run/run_exp1_3_ee_3nodes_cifar10.sh
bash scripts/run/run_exp1_3_ee_3nodes_cifar100.sh
bash scripts/run/run_exp2_homogeneous_multi_model_cifar10.sh
bash scripts/run/run_exp2_homogeneous_multi_model_cifar100.sh
bash scripts/run/run_exp3_memory_aware_multi_model_cifar10.sh
bash scripts/run/run_exp3_memory_aware_multi_model_cifar100.sh

# The distributed scripts still accept explicit config paths for one-off runs.
bash scripts/run/run_exp1_3_ee_3nodes_cifar10.sh configs/experiments/exp1_3_resnet34_ee_3nodes_cifar10.yaml
bash scripts/run/run_exp2_homogeneous_multi_model_cifar100.sh configs/experiments/exp2_resnet18_cifar100.yaml
bash scripts/run/run_exp3_memory_aware_multi_model_cifar10.sh configs/experiments/exp3_2_resnet34_cifar10.yaml
```

## Thesis Artifacts

Generate summaries, tables, and plots:

```bash
bash scripts/run/generate_exp1_thesis_artifacts.sh
bash scripts/run/generate_exp2_thesis_artifacts.sh
bash scripts/run/generate_exp3_thesis_artifacts.sh
```

Each run directory contains `metrics.json`, `inference_times.csv`, and
`resolved_config.json`.
