# Metrics Definition

## Distributed Communication Overhead

Distributed runs record communication overhead as the non-compute portion of the
master-observed inference latency:

```text
communication_overhead_sec = latency_sec - remote_compute_time_sec
```

Where:

- `latency_sec` is the wall-clock time measured by the master for one sample.
- `remote_compute_time_sec` is the sum of the model compute time reported by the
  workers that handled that sample.

This metric includes network transfer time plus protocol/runtime costs such as
serialization, deserialization, HTTP handling, request forwarding, and response
handling. Very small negative values can indicate timing noise between the
master-observed wall-clock measurement and worker-reported compute durations.

Per-sample rows in `latencies.csv` include:

- `communication_overhead_sec`
- `communication_overhead_ratio`

Aggregate fields in `metrics.json` include:

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
