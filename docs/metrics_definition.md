# Metrics Definition

This document defines the main metrics written to `metrics.json`,
`latencies.csv`, and the generated thesis tables.

## Common Metrics

| Metric | Meaning |
| --- | --- |
| `num_samples` | Number of measured samples. In multi-model experiments this is model-sample jobs, so two model instances over 10,000 images becomes 20,000 samples. |
| `num_correct` | Number of correct predictions. |
| `accuracy` | `num_correct / num_samples * 100`. |
| `total_inference_time_sec` | Wall-clock measured interval for the experiment body, excluding setup and result writing. |
| `throughput_samples_per_sec` | `num_samples / total_inference_time_sec`. |
| `avg_latency_sec` | Mean measured latency per sample or model-sample. |
| `p50_latency_sec`, `p95_latency_sec`, `p99_latency_sec` | Latency percentiles. |
| `busy_time_sec` | Sum of measured per-sample latencies. |
| `node_utilization` or `master_node_utilization` | `busy_time_sec / total_inference_time_sec`. For concurrent multi-model runs this can exceed a single sequential worker interpretation because jobs overlap. |

## Energy And Carbon

Single-node runs write:

- `energy_kWh`
- `carbon_kg`

Distributed runs write master, worker, and total fields:

- `master_energy_kWh`
- `master_carbon_kg`
- `<worker_id>_energy_kWh`
- `<worker_id>_carbon_kg`
- `workers_energy_kWh_total`
- `workers_carbon_kg_total`
- `system_energy_kWh_total`
- `system_carbon_kg_total`

These values come from CodeCarbon trackers. They should be treated as
experiment-level estimates, not hardware power-meter readings.

## Distributed Communication Overhead

Distributed runs record communication overhead as the non-compute portion of the
master-observed inference latency:

```text
communication_overhead_sec = latency_sec - remote_compute_time_sec
```

Where:

- `latency_sec` is the wall-clock time measured by the master for one sample or
  model-sample.
- `remote_compute_time_sec` is the sum of model compute time reported by the
  workers that handled that sample.

This metric includes network transfer time plus protocol/runtime costs such as
serialization, deserialization, HTTP handling, request forwarding, and response
handling. Very small negative values can indicate timing noise between the
master-observed wall-clock measurement and worker-reported compute durations.

Per-sample rows in `latencies.csv` include:

- `remote_compute_time_sec`
- `communication_overhead_sec`
- `communication_overhead_ratio`

Aggregate fields in `metrics.json` include:

- `remote_compute_time_total_sec`
- `remote_compute_time_avg_sec`
- `communication_overhead_total_sec`
- `communication_overhead_avg_sec`
- `communication_overhead_std_sec`
- `communication_overhead_min_sec`
- `communication_overhead_max_sec`
- `communication_overhead_p50_sec`
- `communication_overhead_p95_sec`
- `communication_overhead_p99_sec`
- `communication_overhead_ratio_avg`
- `communication_overhead_ratio_total`

## Early-Exit Metrics

Early-exit runs report counts and ratios for four exit IDs:

- `exit_0_count`, `exit_0_ratio`
- `exit_1_count`, `exit_1_ratio`
- `exit_2_count`, `exit_2_ratio`
- `exit_3_count`, `exit_3_ratio`

Exits `0`, `1`, and `2` are intermediate heads. Exit `3` is the final classifier.

For baseline runs, exit fields are present but set to `null` because there are
no early exits.

## Worker Metrics

Distributed runs aggregate per-worker fields:

- `<worker_id>_compute_time_total_sec`
- `<worker_id>_compute_time_avg_sec`
- `<worker_id>_node_utilization`
- `<worker_id>_carbon_kg`
- `<worker_id>_energy_kWh`

Where:

```text
<worker_id>_node_utilization =
    <worker_id>_compute_time_total_sec / total_inference_time_sec
```

The per-sample CSV has matching per-worker compute columns with `_sec` suffixes.

## Multi-Model Metrics

Experiment 2 and the planned Experiment 3 record all distributed metrics plus
model-instance fields:

- `num_workers`
- `num_model_instances`
- `model_instance_ids`
- `samples_per_model`

For each model instance, fields are prefixed with the model ID:

- `model_0_num_samples`
- `model_0_accuracy`
- `model_0_throughput_samples_per_sec`
- `model_0_avg_latency_sec`
- `model_0_p95_latency_sec`
- `model_0_p99_latency_sec`
- `model_0_exit_0_ratio` through `model_0_exit_3_ratio`
- `model_0_<worker_id>_compute_time_total_sec`

The same pattern is repeated for `model_1`, `model_2`, and so on if additional
model instances are configured.

## Experiment 3 Placement Metrics

Experiment 3 is intended to compare a memory-aware placement against the
Experiment 2 placement. In addition to the multi-model metrics above, the
planned metric set should include:

- `placement_strategy`
- `spare_worker_id`
- `route_model_0`
- `route_model_1`
- `<worker_id>_assigned_partitions`
- `<worker_id>_num_stage_0_partitions`
- `<worker_id>_num_stage_1_partitions`
- `<worker_id>_num_stage_2_partitions`
- `<worker_id>_partition_memory_bytes`
- `<worker_id>_partition_memory_mb`
- `max_worker_partition_memory_mb`
- `min_worker_partition_memory_mb`
- `partition_memory_imbalance_ratio`

The primary memory comparison should focus on how many early, high-traffic
partitions each worker stores. Experiment 2 places both first-stage partitions
on `worker1`; Experiment 3 places one first-stage partition on `worker1` and one
first-stage partition on `worker2`.

Experiment 3 per-sample rows should also preserve the chosen route:

- `model_instance_id`
- `route`
- `entry_worker_id`
- `terminal_worker_id`
- `assigned_partition_id`

These fields make it possible to verify that:

- `model_0` follows `worker1 -> worker3 -> worker3`
- `model_1` follows `worker2 -> worker3 -> worker3`
- `worker3` receives the second- and third-stage traffic for both models

## Thesis Tables

The generated table files under `results/thesis_visualizations/**/tables/`
normalize the raw metrics into thesis-friendly views:

- `core_metrics`: accuracy, throughput, latency, communication overhead, and
  sample count.
- `energy_metrics`: master, worker, and total energy/carbon fields.
- `exit_distribution`: exit ratios by experiment.
- `worker_breakdown`: worker compute, utilization, energy, and carbon values.

The combined CSV and JSON files preserve the broader raw metric set for custom
analysis.
