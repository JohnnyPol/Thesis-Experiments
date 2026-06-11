from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_duration_stats(durations: Iterable[float]) -> dict:
    """
    Compute summary statistics for measured durations in seconds.
    """
    durations = list(durations)

    if not durations:
        return {
            "num_samples": 0,
            "avg_duration_sec": 0.0,
            "std_duration_sec": 0.0,
            "min_duration_sec": 0.0,
            "max_duration_sec": 0.0,
            "duration_sum_sec": 0.0,
        }

    arr = np.array(durations, dtype=np.float64)

    return {
        "num_samples": int(arr.size),
        "avg_duration_sec": float(np.mean(arr)),
        "std_duration_sec": float(np.std(arr)),
        "min_duration_sec": float(np.min(arr)),
        "max_duration_sec": float(np.max(arr)),
        "duration_sum_sec": float(np.sum(arr)),
    }


def compute_inference_time_stats(inference_times: Iterable[float]) -> dict:
    """
    Compute per-sample inference-time statistics in seconds.
    """
    stats = compute_duration_stats(inference_times)

    return {
        "num_samples": stats["num_samples"],
        "avg_inference_time_sec": stats["avg_duration_sec"],
        "std_inference_time_sec": stats["std_duration_sec"],
        "min_inference_time_sec": stats["min_duration_sec"],
        "max_inference_time_sec": stats["max_duration_sec"],
        "busy_time_sec": stats["duration_sum_sec"],
    }


def compute_total_inference_time(start_time: float, end_time: float) -> float:
    """
    Compute end-to-end inference runtime for the experiment body.
    """
    return float(end_time - start_time)


def compute_throughput(num_samples: int, total_inference_time_sec: float) -> float:
    """
    Compute throughput in samples/sec.
    """
    if total_inference_time_sec <= 0:
        return 0.0
    return float(num_samples / total_inference_time_sec)
