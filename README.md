# Thesis Experiments: Early-Exit ResNet-18 Inference

This repository contains the experiment code, configs, run scripts, metrics, and
thesis visualization pipeline for evaluating ResNet-18 inference across
single-node, distributed single-model, heterogeneous, and distributed
multi-model deployments.

The main research comparison has two implemented experiment groups and one
final defined experiment design:

- **Experiment 1**: single-model inference across baseline, early-exit,
  homogeneous distributed, and heterogeneous distributed topologies.
- **Experiment 2**: homogeneous distributed early-exit inference where `N-1`
  logical model instances share an `N`-worker pipeline.
- **Experiment 3**: memory-aware multi-model partition placement for two
  early-exit ResNet-18 model instances on three workers, with the two
  high-traffic first partitions split across `worker1` and `worker2`, and
  `worker3` used for the later partitions.

`configs/experiments/exp3.yaml` contains the intended Experiment 3 placement
and routing metadata, but the runtime changes for per-model routing are not yet
implemented in the current codebase.

## Experiment Catalog

| ID | Name | Model | Topology | Entrypoint | Output |
| --- | --- | --- | --- | --- | --- |
| `exp1_1` | Baseline single node | ResNet-18 baseline | one CPU worker | `src.inference.single_node` | `results/exp1_single_model/01_single_node_baseline/run_001/` |
| `exp1_2` | Early-exit single node | EE ResNet-18 | one CPU worker | `src.inference.single_node` | `results/exp1_single_model/02_single_node_ee/run_001/` |
| `exp1_3` | EE homogeneous 2 workers | EE ResNet-18 | two Raspberry Pi CPU workers | `src.distributed.master_client` | `results/exp1_single_model/03_homogeneous_2workers_ee/run_001/` |
| `exp1_4` | EE homogeneous 3 workers | EE ResNet-18 | three Raspberry Pi CPU workers | `src.distributed.master_client` | `results/exp1_single_model/04_homogeneous_3workers_ee/run_001/` |
| `exp1_5` | EE heterogeneous 2 workers | EE ResNet-18 | Raspberry Pi CPU + Jetson GPU final stage | `src.distributed.master_client` | `results/exp1_single_model/05_heterogeneous_pi_jetson_2workers/run_001/` |
| `exp1_6` | EE heterogeneous 3 workers | EE ResNet-18 | two Raspberry Pi CPU workers + Jetson GPU final stage | `src.distributed.master_client` | `results/exp1_single_model/06_heterogeneous_2pis_jetson_3workers/run_001/` |
| `exp2` | Homogeneous multi-model EE | EE ResNet-18 | three homogeneous CPU workers, two logical models by default | `src.distributed.multi_model_master_client` | `results/exp2_multi_model/01_homogeneous_3workers_2models/run_001/` |
| `exp3` | Memory-aware multi-model placement | EE ResNet-18 | two logical models, three workers, first partitions split across `worker1` and `worker2`, later partitions on `worker3` | planned per-model routing master/runtime | `results/exp3_memory_aware_multi_model/01_3workers_2models_first_split_late_consolidated/run_001/` |

## What Is Measured

Every run writes machine-readable metrics and per-sample latency records. The
metrics include:

- accuracy, correct predictions, and sample count
- total inference time, throughput, average latency, p50, p95, and p99 latency
- CodeCarbon energy and carbon estimates
- network byte deltas on the configured interface
- early-exit counts and ratios for exits `0`, `1`, `2`, and final exit `3`

Distributed runs additionally measure:

- master-observed protocol bytes
- worker request and response byte estimates
- worker compute time totals and averages
- remote compute time
- communication overhead, computed as `latency_sec - remote_compute_time_sec`
- worker-level CodeCarbon energy and carbon estimates
- total system energy and carbon estimates across master and workers

Experiment 2 additionally records:

- number of model instances
- model instance IDs, such as `model_0,model_1`
- samples per model
- per-model accuracy, latency, throughput, exit distribution, and worker compute
  totals

Experiment 3 is intended to add placement-specific comparison metrics:

- assigned partitions per worker
- first-, second-, and third-stage partition counts per worker
- per-worker model memory footprint or parameter-memory estimate
- utilization and compute balance relative to Experiment 2
- communication overhead introduced by cross-model routing

For metric definitions and interpretation notes, see
[`docs/metrics_definition.md`](docs/metrics_definition.md). For the detailed
experiment mechanics, see [`docs/experiment_guide.md`](docs/experiment_guide.md).

## Repository Layout

```text
configs/
  datasets/              Dataset configs used by the experiment bundle loader.
  experiments/           Experiment-level configs and output locations.
  models/                Baseline and early-exit ResNet-18 configs.
  systems/               Single-node, homogeneous, and heterogeneous systems.
docs/
  experiment_guide.md    Detailed explanation of all experiments and runtime flow.
  jetson_legacy_runtime.md
  metrics_definition.md
scripts/
  deploy/                Worker setup, sync, service, and log collection helpers.
  run/                   Experiment and artifact-generation scripts.
src/
  data/                  CIFAR-10 loaders and transforms.
  distributed/           FastAPI worker service, master clients, protocol, runtime.
  distributed_legacy/    Python 3.6-compatible Jetson worker runtime.
  inference/             Single-node and partitioned inference entrypoints.
  metrics/               Accuracy, latency, network, energy, exit, utilization helpers.
  models/                Baseline ResNet, EE ResNet, exit blocks, partitions.
  visualization/         Thesis summary, table, and plot generation.
results/
  exp1_single_model/     Raw Experiment 1 run outputs.
  exp2_multi_model/      Raw Experiment 2 run outputs.
  exp3_memory_aware_multi_model/
                          Planned raw Experiment 3 run outputs.
  thesis_visualizations/ Combined CSV/JSON, markdown tables, and plots.
```

Important entrypoints:

- [`src/inference/single_node.py`](src/inference/single_node.py): single-node
  baseline and single-node early-exit inference.
- [`src/distributed/api/app.py`](src/distributed/api/app.py): FastAPI worker
  service for distributed inference.
- [`src/distributed/master_client.py`](src/distributed/master_client.py):
  distributed single-model master coordinator.
- [`src/distributed/multi_model_master_client.py`](src/distributed/multi_model_master_client.py):
  Experiment 2 multi-model master coordinator.
- [`src/models/partitioning.py`](src/models/partitioning.py): supported 2-way
  and 3-way EE ResNet-18 partition modules.
- [`src/visualization/summary.py`](src/visualization/summary.py),
  [`src/visualization/tables.py`](src/visualization/tables.py), and
  [`src/visualization/plots.py`](src/visualization/plots.py): thesis artifacts.

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the repository dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch and TorchVision separately when needed for Raspberry Pi CPU
execution:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://www.piwheels.org/simple \
  torch

pip install torchvision --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://www.piwheels.org/simple
```

Set the project root on `PYTHONPATH` before running modules manually:

```bash
export PYTHONPATH="$(pwd)"
```

PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## Data, Models, And Checkpoints

The implemented experiment configs use:

- dataset: CIFAR-10 test split
- image preprocessing: resize to `256x256`, convert to tensor, normalize with
  CIFAR-10 mean and standard deviation
- batch size: `1`
- device: CPU for Raspberry Pi and single-node runs; Jetson configs use `gpu`
  for the final worker
- baseline checkpoint: `checkpoints/resnet18_baseline.pth`
- early-exit checkpoint: `checkpoints/resnet18_ee_entropy.pth`

The CIFAR-10 dataset config has `download: true`, so the dataset can be
downloaded automatically into `./data` if it is missing. For distributed runs,
make sure every worker has the same repository state, Python dependencies,
dataset files, and checkpoint files.

The early-exit model has three intermediate exits plus the final classifier:

- exit `0`: after `layer0`
- exit `1`: after `layer1`
- exit `2`: after `layer2`
- exit `3`: final classifier after `layer3`, adaptive average pooling, and `fc`

The current early-exit policy is entropy-based. With
`confidence_threshold: 0.9`, a sample exits early when entropy is less than or
equal to `0.9`.

## Configuration Model

Each experiment config composes three referenced configs:

```yaml
config_refs:
  dataset: configs/datasets/cifar10.yaml
  model: configs/models/resnet18_ee_entropy.yaml
  system: configs/systems/homogeneous_3workers.yaml
```

`src.utils.config.load_experiment_bundle()` resolves those paths relative to the
repository root and saves the fully resolved bundle as `resolved_config.json`
inside each run directory.

Common controls:

- `runtime.batch_size`: currently expected to be `1` for early-exit and
  distributed inference.
- `runtime.warmup_samples`: samples run before measurement.
- `runtime.save_per_sample_metrics`: records `latencies.csv`.
- `runtime.model_instance_count`: Experiment 2 logical model count; `auto`
  resolves to `N-1` for `N` workers. Experiment 3 uses exactly `2` logical
  models on `3` workers.
- `runtime.concurrency`: Experiment 2 and Experiment 3 thread-pool concurrency.
- `runtime.max_samples_per_model`: multi-model sample cap per logical model;
  `null` means the full test set.
- `placement.assignments`: Experiment 3 worker-to-model-stage placement table.
- `placement.routes`: Experiment 3 per-model routing table.
- `output.dir`: run output directory.
- `system.pipeline_order`: worker execution order.
- `system.workers[*].connect_host` and `port`: addresses used by the master and
  upstream workers.
- `system.monitoring.network_interface`: interface used for network byte deltas.

## Running Experiment 1

### Single-Node Baseline

```bash
bash scripts/run/run_exp1_1_baseline_single_node.sh
```

Equivalent module command:

```bash
python -m src.inference.single_node \
  --config configs/experiments/exp1_1_baseline_single_node.yaml
```

### Single-Node Early Exit

```bash
bash scripts/run/run_exp1_2_ee_single_node.sh
```

Equivalent module command:

```bash
python -m src.inference.single_node \
  --config configs/experiments/exp1_2_ee_single_node.yaml
```

### Distributed Single-Model Runs

Start a worker API on each worker node. Use the same experiment config as the
run you are about to execute and pass the worker ID from the selected system
config:

```bash
bash scripts/run/start_worker_api.sh configs/experiments/exp1_4_ee_homogeneous_3nodes.yaml worker1
bash scripts/run/start_worker_api.sh configs/experiments/exp1_4_ee_homogeneous_3nodes.yaml worker2
bash scripts/run/start_worker_api.sh configs/experiments/exp1_4_ee_homogeneous_3nodes.yaml worker3
```

Check worker health from the master:

```bash
curl http://192.168.0.102:9101/health
curl http://192.168.0.104:9102/health
curl http://192.168.0.103:9103/health
```

Then run the matching master script:

```bash
bash scripts/run/run_exp1_3_ee_homogeneous_2nodes.sh
bash scripts/run/run_exp1_4_ee_homogeneous_3nodes.sh
bash scripts/run/run_exp1_5_ee_heterogeneous.sh
bash scripts/run/run_exp1_6_ee_heterogeneous_3nodes.sh
```

For Jetson Nano workers running JetPack 4.x / Python 3.6, use the legacy worker
runtime described in [`docs/jetson_legacy_runtime.md`](docs/jetson_legacy_runtime.md).

## Running Experiment 2

Experiment 2 uses:

```text
configs/experiments/exp2.yaml
configs/systems/homogeneous_3workers.yaml
configs/models/resnet18_ee_entropy.yaml
configs/datasets/cifar10.yaml
```

The default 3-worker system creates `N-1 = 2` logical model instances:

```text
model_0
model_1
```

With `max_samples_per_model: null`, each logical model runs the full CIFAR-10
test set:

```text
10,000 samples per model x 2 models = 20,000 model-sample inferences
```

Start one worker service per worker:

```bash
bash scripts/run/start_worker_api.sh configs/experiments/exp2.yaml worker1
bash scripts/run/start_worker_api.sh configs/experiments/exp2.yaml worker2
bash scripts/run/start_worker_api.sh configs/experiments/exp2.yaml worker3
```

Run the Experiment 2 master:

```bash
bash scripts/run/run_exp2_homogeneous_multi_model.sh
```

That script checks every configured worker `/health` endpoint before running:

```bash
python -m src.distributed.multi_model_master_client \
  --config configs/experiments/exp2.yaml
```

## Experiment 3 Design

Experiment 3 is the final planned experiment. It keeps the same physical cluster
size as Experiment 2:

```text
3 workers
2 logical early-exit ResNet-18 model instances
3 partitions per model
worker3 = consolidated later-partition worker
```

The motivation is to keep the high-traffic first partitions off a single
Raspberry Pi. In Experiment 2, the placement is stage-parallel for every model:

```text
worker1:
  model_0 stage_0
  model_1 stage_0

worker2:
  model_0 stage_1
  model_1 stage_1

worker3:
  model_0 stage_2
  model_1 stage_2
```

That means `worker1` holds two first-stage partitions. Those partitions see all
input samples and are expected to be the most frequently executed stages.

Experiment 3 uses a memory-aware cross-model placement:

```text
worker1:
  model_0 stage_0

worker2:
  model_1 stage_0

worker3:
  model_0 stage_1
  model_0 stage_2
  model_1 stage_1
  model_1 stage_2
```

This placement enforces the core constraints:

- each Raspberry Pi handles at most one first-stage model partition
- `worker3` manages the deeper second- and third-stage partitions that receive
  fewer samples because earlier exits progressively reduce traffic

The resulting per-model routes are:

```text
model_0: worker1 stage_0 -> worker3 stage_1 -> worker3 stage_2
model_1: worker2 stage_0 -> worker3 stage_1 -> worker3 stage_2
```

The planned thesis comparison is against Experiment 2. The expected effect is a
lower first-stage traffic concentration on `worker1` and explicit use of
`worker3` for later, lower-traffic work. The tradeoff to measure is additional
traffic to `worker3` and its impact on communication overhead.

The current runtime still assumes a static worker pipeline for each system
config unless an experiment defines a `placement` block. Experiment 3 enables
placement-aware routing, where the next worker is chosen from
`(model_instance_id, current_stage_id)` instead of a single static
`next_worker_id`.

An intended config shape is:

```yaml
experiment:
  id: exp3
  name: memory_aware_multi_model_placement

runtime:
  batch_size: 1
  warmup_samples: 20
  model_instance_count: 2
  concurrency: 2
  max_samples_per_model: null
  save_per_sample_metrics: true
  save_predictions: false
  placement_strategy: explicit_first_partition_split

placement:
  spare_worker_id: worker3
  assignments:
    worker1:
      - model_instance_id: model_0
        partition_id: 0
    worker2:
      - model_instance_id: model_1
        partition_id: 0
    worker3:
      - model_instance_id: model_0
        partition_id: 1
      - model_instance_id: model_0
        partition_id: 2
      - model_instance_id: model_1
        partition_id: 1
      - model_instance_id: model_1
        partition_id: 2
  routes:
    model_0:
      - worker_id: worker1
        partition_id: 0
      - worker_id: worker3
        partition_id: 1
      - worker_id: worker3
        partition_id: 2
    model_1:
      - worker_id: worker2
        partition_id: 0
      - worker_id: worker3
        partition_id: 1
      - worker_id: worker3
        partition_id: 2
  constraints:
    max_first_partitions_per_worker: 1
    later_partitions_worker_id: worker3
```

Start one worker service per worker:

```bash
bash scripts/run/start_worker_api.sh configs/experiments/exp3.yaml worker1
bash scripts/run/start_worker_api.sh configs/experiments/exp3.yaml worker2
bash scripts/run/start_worker_api.sh configs/experiments/exp3.yaml worker3
```

Run the Experiment 3 master:

```bash
bash scripts/run/run_exp3_memory_aware_multi_model.sh
```

## How Distributed Inference Works

1. The master loads the CIFAR-10 test set and sends each image tensor to the
   first worker through `POST /infer`.
2. The tensor is serialized as multipart form data with JSON metadata and raw
   tensor bytes.
3. The worker executes its local partition.
4. If an early exit fires, the worker returns the prediction to the master.
5. If no exit fires and another worker exists, the worker forwards the
   activation tensor to the next worker.
6. The terminal response is enriched with stage metrics from every worker on
   the path.
7. The master records latency, prediction, exit ID, path, protocol bytes,
   remote compute time, communication overhead, and worker byte/compute fields.

Supported distributed partition counts are currently `2` and `3`. Experiment 3
uses the existing 3-partition model structure, but needs dynamic per-model route
selection rather than one shared route for all model instances.

## Output Files

Each run directory contains:

- `metrics.json`: aggregate metrics and experiment metadata.
- `latencies.csv`: per-sample or per-model-sample records.
- `resolved_config.json`: resolved experiment, dataset, model, and system
  configuration bundle.

Important `latencies.csv` columns include:

- `sample_index`, `latency_sec`, `predicted_class`, `true_class`, `correct`,
  `exit_id`
- distributed only: `confidence`, `protocol_bytes`,
  `remote_compute_time_sec`, `communication_overhead_sec`,
  `communication_overhead_ratio`, `path`
- Experiment 2 only: `model_instance_id`
- Experiment 3 planned: placement metadata such as `route`, assigned
  partition, and per-worker partition-memory fields
- per-worker distributed fields such as `worker1_compute_time_sec`,
  `worker1_request_bytes`, and `worker1_response_bytes`

## Thesis Tables And Plots

Generate Experiment 1 thesis artifacts:

```bash
bash scripts/run/generate_exp1_thesis_artifacts.sh
```

Generate Experiment 2 thesis artifacts:

```bash
bash scripts/run/generate_exp2_thesis_artifacts.sh
```

Experiment 3 should use the same artifact pipeline once raw run outputs exist,
with Experiment 3-specific input and output directories:

```bash
bash scripts/run/generate_exp2_thesis_artifacts.sh \
  results/exp3_memory_aware_multi_model \
  results/thesis_visualizations/exp3_memory_aware_multi_model
```

PowerShell:

```powershell
.\scripts\run\generate_exp1_thesis_artifacts.ps1
.\scripts\run\generate_exp2_thesis_artifacts.ps1
```

The artifact pipeline writes:

- `combined_metrics.csv`
- `combined_metrics.json`
- `experiment_overview.md`
- `tables/core_metrics.csv` and `.md`
- `tables/energy_metrics.csv` and `.md`
- `tables/exit_distribution.csv` and `.md`
- `tables/worker_breakdown.csv` and `.md`
- `plots/performance_overview.png`
- `plots/energy_emissions.png`
- `plots/distributed_network_protocol.png`
- `plots/distributed_communication_overhead.png`
- `plots/exit_distribution.png`
- `plots/worker_compute_breakdown.png`

Custom input and output directories are supported:

```bash
bash scripts/run/generate_exp2_thesis_artifacts.sh \
  results/exp2_multi_model \
  results/thesis_visualizations/exp2_multi_model
```

```powershell
.\scripts\run\generate_exp2_thesis_artifacts.ps1 `
  results/exp2_multi_model `
  results/thesis_visualizations/exp2_multi_model
```

## Current Result Summaries

Generated thesis overview files are available at:

- [`results/thesis_visualizations/exp1_single_model/experiment_overview.md`](results/thesis_visualizations/exp1_single_model/experiment_overview.md)
- [`results/thesis_visualizations/exp2_multi_model/experiment_overview.md`](results/thesis_visualizations/exp2_multi_model/experiment_overview.md)

The current checked/generated summaries report:

- Experiment 1 includes `exp1_1` through `exp1_6`.
- Experiment 2 includes the homogeneous 3-worker, 2-model run.
- Experiment 3 is documented as the final memory-aware placement experiment,
  but no raw `exp3` result summary exists yet.
- The best throughput in the current Experiment 1 summary is `exp1_2`
  single-node early exit.
- The current Experiment 2 summary reports throughput around `3.117`
  model-samples/sec for the homogeneous multi-model run.

Regenerate the thesis artifacts whenever raw `metrics.json` or `latencies.csv`
files change.

## Limitations And Notes

- `src.inference.single_node.main()` currently passes `max_samples=1` into
  `evaluate_single_node()`. The saved single-node results in this workspace may
  have been produced by a different local version or direct function call if
  they contain full-dataset counts.
- Distributed master clients are designed for full-dataset evaluation when no
  sample cap is configured.
- Distributed EE inference supports only `batch_size=1`.
- Distributed partitioning currently supports 2-worker and 3-worker topologies.
- Experiment 2 validates that the number of logical model instances equals
  `N-1` for `N` workers.
- Experiment 3 uses the `placement` block for per-model route lookup,
  per-worker multi-partition loading, and placement-aware route metrics.
- Protocol byte accounting is a stable estimate built from tensor payload sizes,
  metadata size, headers, and fixed HTTP/multipart overhead constants; it is not
  packet-capture data.
- CodeCarbon estimates depend on host platform support and current system load.

## Reproducibility Checklist

- Use the same checkpoint files for all comparisons.
- Keep the same dataset root and preprocessing config on every node.
- Keep `batch_size=1` for early-exit and distributed experiments.
- Verify every worker `/health` endpoint before starting a distributed master.
- Keep worker IP addresses, ports, and `pipeline_order` synchronized.
- Keep the configured network interface stable when comparing network bytes.
- Run compared experiments under similar system load.
- Record package versions for `torch`, `torchvision`, `numpy`, `pandas`,
  `matplotlib`, `fastapi`, `uvicorn`, and `codecarbon`.
- Regenerate thesis artifacts after rerunning experiments.
