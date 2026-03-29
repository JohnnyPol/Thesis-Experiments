from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_latency_stats(latencies: Iterable[float]) -> dict:
    """
    Compute summary latency statistics from per-sample latencies.

    Args:
        latencies: Iterable of latency values in seconds.

    Returns:
        Dictionary with mean/std/min/max/p50/p95/p99 and count.
    """
    latencies = list(latencies)

    if not latencies:
        return {
            "num_samples": 0,
            "avg_latency_sec": 0.0,
            "std_latency_sec": 0.0,
            "min_latency_sec": 0.0,
            "max_latency_sec": 0.0,
            "p50_latency_sec": 0.0,
            "p95_latency_sec": 0.0,
            "p99_latency_sec": 0.0,
            "busy_time_sec": 0.0,
        }

    arr = np.array(latencies, dtype=np.float64)

    return {
        "num_samples": int(arr.size),
        "avg_latency_sec": float(np.mean(arr)),
        "std_latency_sec": float(np.std(arr)),
        "min_latency_sec": float(np.min(arr)),
        "max_latency_sec": float(np.max(arr)),
        "p50_latency_sec": float(np.percentile(arr, 50)),
        "p95_latency_sec": float(np.percentile(arr, 95)),
        "p99_latency_sec": float(np.percentile(arr, 99)),
        "busy_time_sec": float(np.sum(arr)),
    }


def compute_total_inference_time(start_time: float, end_time: float) -> float:
    """
    Compute end-to-end inference runtime.

    Args:
        start_time: Experiment start timestamp.
        end_time: Experiment end timestamp.

    Returns:
        Total inference time in seconds.
    """
    return float(end_time - start_time)


def compute_throughput(num_samples: int, total_inference_time_sec: float) -> float:
    """
    Compute throughput in samples/sec.

    Args:
        num_samples: Number of processed samples.
        total_inference_time_sec: End-to-end experiment runtime in seconds.

    Returns:
        Throughput in samples/sec.
    """
    if total_inference_time_sec <= 0:
        return 0.0
    return float(num_samples / total_inference_time_sec)