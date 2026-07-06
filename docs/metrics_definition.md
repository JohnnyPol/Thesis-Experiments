# Metrics Definition

This reference defines metrics produced by current inference code and legacy names found in stored CIFAR-10 results. An **inference task** is one `(input image, logical model instance)` pair.

## Files and schema versions

Current runs write `metrics.json`, `inference_times.csv`, and `resolved_config.json`. Archived CIFAR-10 runs predate a naming cleanup:

| Meaning | Archived CIFAR-10 | Current/CIFAR-10.1 |
| --- | --- | --- |
| Per-task CSV | `latencies.csv` | `inference_times.csv` |
| Per-task time | `latency_sec` | `inference_time_sec` |
| Mean | `avg_latency_sec` | `avg_inference_time_sec` |
| Spread/range | `std/min/max_latency_sec` | `std/min/max_inference_time_sec` |
| Per-instance mean | `<model>_avg_latency_sec` | `<model>_avg_inference_time_sec` |

The table and plot modules explicitly accept both aggregate schemas.

## Task counts and accuracy

| Field | Definition |
| --- | --- |
| `num_samples` | Number of measured tasks |
| `num_correct` | Correctly classified tasks |
| `accuracy` | `100 × num_correct / num_samples` |

Single-model task count equals test-image count. With two instances, CIFAR-10 produces 20,000 tasks and CIFAR-10.1 produces 4,000.

## Timing and throughput

For task `j`, `t_j` is local model time in single-node mode and master-observed end-to-end time in distributed mode.

| Field | Definition |
| --- | --- |
| `avg_inference_time_sec` | Mean of `t_j` |
| `std_inference_time_sec` | Population standard deviation |
| `min_inference_time_sec`, `max_inference_time_sec` | Observed range |
| `busy_time_sec` | `Σ_j t_j` |
| `total_inference_time_sec` | Measured wall-clock experiment body, excluding warm-up |
| `throughput_samples_per_sec` | `num_samples / total_inference_time_sec` |

Busy time sums task intervals; total inference time is wall clock. They differ under concurrency.

## Early-exit metrics

| Exit | Location | Fields |
| --- | --- | --- |
| `0` | After `layer0` | `exit_0_count`, `exit_0_ratio` |
| `1` | After `layer1` | `exit_1_count`, `exit_1_ratio` |
| `2` | After `layer2` | `exit_2_count`, `exit_2_ratio` |
| `3` | Final classifier | `exit_3_count`, `exit_3_ratio` |

Each ratio is `exit_i_count / num_samples`. Baseline files keep these fields as `null`. Partition 0 runs for every task; later partitions run only for tasks that reject earlier exits.

## Utilization

```text
node_utilization = busy_time_sec / total_inference_time_sec

<worker>_node_utilization =
    <worker>_compute_time_total_sec / total_inference_time_sec
```

Distributed master utilization uses master-observed task times. Worker utilization is **application-level normalized compute load**, not operating-system CPU utilization. Concurrent compute intervals can overlap, so values can exceed 100%.

Worker fields include compute totals/means, utilization, energy/carbon, and—for Experiment 3—assigned partitions and counts of assigned stages 0/1/2.

## Remote compute and communication/runtime overhead

```text
remote_compute_time_sec = Σ partition_compute_time_sec

communication_overhead_sec =
    inference_time_sec - remote_compute_time_sec
```

Overhead includes tensor serialization/deserialization, HTTP/FastAPI handling, LAN transfer, forwarding, response handling, synchronization, and queueing. It is not a packet-level measurement or byte counter. Very small negative values can result from timing noise.

Aggregate fields include remote-compute total/mean; overhead total/mean/standard deviation/range; `communication_overhead_ratio_avg`; and:

```text
communication_overhead_ratio_total = Σ overhead_j / Σ t_j
```

This denominator is summed task latency, not wall-clock run duration.

## Energy and carbon

Single-node runs write `energy_kWh` and `carbon_kg`. Distributed runs write master, per-worker, workers-total, and system-total variants. System energy is master energy plus all worker energy.

Thesis visualizations derive:

```text
energy_per_task_J = system_energy_kWh_total × 3.6e6 / num_samples
EDP = energy_per_task_J × mean_inference_time_sec
```

CodeCarbon values are software estimates, not external power-meter readings. They are intended mainly for relative comparisons on the same cluster, not absolute laboratory measurements.

## Multi-model fields

Experiment 2/3 add `num_workers`, `num_model_instances`, `model_instance_ids`, and `samples_per_model`. Each instance has prefixed counts, accuracy, throughput, timing, exit, and per-worker compute fields such as `model_0_accuracy` and `model_0_exit_0_ratio`.

Per-task rows add `model_instance_id`, `path`, `entry_worker_id`, `terminal_worker_id`, `assigned_partition_id`, and per-worker compute columns.

## Experiment 3 placement fields

Experiment 3 is implemented. Stored aggregates include:

- `placement_strategy` and `spare_worker_id` where applicable;
- `route_model_0`, `route_model_1`;
- `<worker>_assigned_partitions`;
- `<worker>_num_stage_0_partitions`, `_num_stage_1_partitions`, and `_num_stage_2_partitions`.

The code does **not** emit the partition-memory byte/MB and memory-imbalance fields described in an earlier draft. “Memory-aware” identifies the static placement rationale; the final evaluation measures timing, throughput, utilization, overhead, and energy consequences rather than a stored byte-level memory metric.

## Generated artifacts

`src.visualization.tables` emits CSV/LaTeX; `src.visualization.plots` emits PNG/PDF. Dataset-aware directories under `results/thesis_visualizations/` contain Experiment 1, Experiment 2/3, and energy views for CIFAR-10 and CIFAR-10.1.
