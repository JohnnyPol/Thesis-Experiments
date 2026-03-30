from __future__ import annotations

from typing import Iterable


def initialize_exit_counts(num_exits: int) -> dict[int, int]:
    """
    Create a zero-initialized exit counter dictionary.

    Example for num_exits=4:
        {0: 0, 1: 0, 2: 0, 3: 0}
    """
    if num_exits <= 0:
        raise ValueError("num_exits must be positive")

    return {exit_id: 0 for exit_id in range(num_exits)}


def update_exit_counts(
    exit_counts: dict[int, int],
    exit_id: int,
) -> dict[int, int]:
    """
    Increment the counter for the selected exit.
    """
    if exit_id not in exit_counts:
        exit_counts[exit_id] = 0

    exit_counts[exit_id] += 1
    return exit_counts


def compute_exit_distribution(
    exit_counts: dict[int, int],
    total_samples: int | None = None,
) -> dict[int, float]:
    """
    Compute normalized exit distribution.

    Returns:
        dict mapping exit_id -> fraction in [0, 1]
    """
    if total_samples is None:
        total_samples = sum(exit_counts.values())

    if total_samples <= 0:
        return {exit_id: 0.0 for exit_id in sorted(exit_counts.keys())}

    return {
        exit_id: exit_counts.get(exit_id, 0) / total_samples
        for exit_id in sorted(exit_counts.keys())
    }


def summarize_exit_counts(
    exit_counts: dict[int, int],
    total_samples: int | None = None,
) -> dict[str, int | float]:
    """
    Flatten exit counts + ratios into a JSON-friendly summary dict.

    Example keys:
        exit_0_count
        exit_0_ratio
        exit_1_count
        exit_1_ratio
        ...
    """
    if total_samples is None:
        total_samples = sum(exit_counts.values())

    distribution = compute_exit_distribution(exit_counts, total_samples)
    summary: dict[str, int | float] = {}

    for exit_id in sorted(exit_counts.keys()):
        summary[f"exit_{exit_id}_count"] = int(exit_counts[exit_id])
        summary[f"exit_{exit_id}_ratio"] = float(distribution[exit_id])

    return summary


def infer_num_exits_from_ids(exit_ids: Iterable[int]) -> int:
    """
    Infer number of exits from observed exit ids.

    If exit ids are [0,1,3], returns 4.
    """
    exit_ids = list(exit_ids)
    if not exit_ids:
        return 0
    return max(exit_ids) + 1