from __future__ import annotations


def compute_node_utilization(time_busy_sec: float, total_inference_time_sec: float) -> float:
    """
    Compute application-level node utilization.

    U_N = time_busy / total_inference_time

    Args:
        time_busy_sec: Sum of per-sample forward-pass times.
        total_inference_time_sec: End-to-end experiment runtime.

    Returns:
        Utilization ratio in [0, 1] when measured consistently.
    """
    if total_inference_time_sec <= 0:
        return 0.0
    return float(time_busy_sec / total_inference_time_sec)