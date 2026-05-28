# Experiment Guide

This guide describes the active experiment set and runtime conventions.

Each experiment config filename and experiment ID explicitly includes the dataset
suffix: `_cifar10` or `_cifar100`. Result directories use the same suffix so
runs cannot be mixed accidentally.

## Experiment 1

| ID | Config | Description |
| --- | --- | --- |
| `exp1_1_resnet18_cifar10` | `configs/experiments/exp1_1_resnet18_baseline_single_node_cifar10.yaml` | Single-node ResNet-18 baseline. |
| `exp1_1_resnet34_cifar10` | `configs/experiments/exp1_1_resnet34_baseline_single_node_cifar10.yaml` | Single-node ResNet-34 baseline. |
| `exp1_2_resnet18_cifar10` | `configs/experiments/exp1_2_resnet18_ee_single_node_cifar10.yaml` | Single-node ResNet-18 early-exit model. |
| `exp1_2_resnet34_cifar10` | `configs/experiments/exp1_2_resnet34_ee_single_node_cifar10.yaml` | Single-node ResNet-34 early-exit model. |
| `exp1_3_resnet18_cifar10` | `configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10.yaml` | Three-worker ResNet-18 early-exit model. |
| `exp1_3_resnet34_cifar10` | `configs/experiments/exp1_3_resnet34_ee_3nodes_cifar10.yaml` | Three-worker ResNet-34 early-exit model. |

## Experiment 2

Experiment 2 runs two logical models on three workers:

- `configs/experiments/exp2_resnet18_cifar10.yaml`: ResNet-18 EE / ResNet-18 EE.
- `configs/experiments/exp2_resnet34_cifar10.yaml`: ResNet-34 EE / ResNet-34 EE.
- `configs/experiments/exp2_resnet18_cifar100.yaml`: CIFAR-100 ResNet-18 EE / ResNet-18 EE.
- `configs/experiments/exp2_resnet34_cifar100.yaml`: CIFAR-100 ResNet-34 EE / ResNet-34 EE.

Both use `model_instance_count: auto`, which resolves to `N-1` logical models
for `N` workers. With three workers this creates `model_0` and `model_1`.

## Experiment 3

Experiment 3 has three placement arrangements, each available for two ResNet-18
models or two ResNet-34 models:

| ID | Config | Placement |
| --- | --- | --- |
| `exp3_1_resnet18_cifar10` | `configs/experiments/exp3_1_resnet18_cifar10.yaml` | First partitions split, later partitions consolidated on `worker3`. |
| `exp3_1_resnet34_cifar10` | `configs/experiments/exp3_1_resnet34_cifar10.yaml` | First partitions split, later partitions consolidated on `worker3`. |
| `exp3_2_resnet18_cifar10` | `configs/experiments/exp3_2_resnet18_cifar10.yaml` | First partitions split, one second-stage partition staggered onto `worker2`. |
| `exp3_2_resnet34_cifar10` | `configs/experiments/exp3_2_resnet34_cifar10.yaml` | First partitions split, one second-stage partition staggered onto `worker2`. |
| `exp3_3_resnet18_cifar10` | `configs/experiments/exp3_3_resnet18_cifar10.yaml` | First partitions split, later partitions cross-balanced across all workers. |
| `exp3_3_resnet34_cifar10` | `configs/experiments/exp3_3_resnet34_cifar10.yaml` | First partitions split, later partitions cross-balanced across all workers. |

The 3.1 routes are:

```text
model_0: worker1 stage_0 -> worker3 stage_1 -> worker3 stage_2
model_1: worker2 stage_0 -> worker3 stage_1 -> worker3 stage_2
```

## Partitioning

The distributed partition IDs are:

```text
stage_0: conv1 -> maxpool -> layer0 -> exit0 -> layer1 -> exit1
stage_1: layer2 -> exit2
stage_2: layer3 -> avgpool -> fc
```

## Models

Model architecture is read from `configs/models/*.yaml`:

```yaml
architecture: resnet18
```

Supported early-exit architectures are `resnet18` and `resnet34`. Both use the
same exit locations:

- exit `0`: after `layer0`
- exit `1`: after `layer1`
- exit `2`: after `layer2`
- exit `3`: final classifier after `layer3`, adaptive average pooling, and `fc`

## Runtime

Single-node inference is handled by `src.inference.single_node`.

Distributed inference uses:

- `src.distributed.api.app`: worker FastAPI service.
- `src.distributed.master_client`: single-model distributed master.
- `src.distributed.multi_model_master_client`: multi-model distributed master.
- `src.models.partitioning`: local partition modules.
- `src.distributed.runtime.worker_runtime`: worker placement and route state.

Distributed workers are config-specific: changing from ResNet-18 CIFAR-10 to
ResNet-18 CIFAR-100, or to ResNet-34, requires restarting each worker API with
the matching experiment config before launching the master. The run scripts
validate `/info` so a master run cannot silently use workers loaded with another
model or dataset.

Active system configs:

- `configs/systems/single_node_worker1.yaml`
- `configs/systems/homogeneous_3workers.yaml`

Each experiment config composes a dataset config, model config, and system
config. The resolved bundle is written to `resolved_config.json` inside the run
directory.

CIFAR-100 model configs point at separate 100-class checkpoint files under
`checkpoints/*_cifar100.pth`, so CIFAR-10 classifiers are not accidentally
loaded into CIFAR-100 model heads.
