from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("results/exp1_single_model")
DEFAULT_OUTPUT_DIR = Path("results/thesis_visualizations/exp1_single_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined summary dataset for Experiment 1 thesis outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Root directory that contains experiment run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where summary artifacts will be written.",
    )
    return parser.parse_args()


def load_metrics_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/run_*/metrics.json"))


def _compute_communication_overhead_fields(metrics_path: Path) -> dict[str, float]:
    latencies_path = metrics_path.with_name("latencies.csv")
    if not latencies_path.exists():
        return {}

    df = pd.read_csv(latencies_path)
    required_columns = {"latency_sec", "remote_compute_time_sec"}
    if not required_columns.issubset(df.columns):
        return {}

    overhead = (
        df["latency_sec"].astype(float) - df["remote_compute_time_sec"].astype(float)
    ).to_numpy(dtype=np.float64)
    latency = df["latency_sec"].astype(float).to_numpy(dtype=np.float64)
    ratios = np.divide(
        overhead,
        latency,
        out=np.zeros_like(overhead, dtype=np.float64),
        where=latency > 0.0,
    )

    if overhead.size == 0:
        return {
            "communication_overhead_total_sec": 0.0,
            "communication_overhead_avg_sec": 0.0,
            "communication_overhead_std_sec": 0.0,
            "communication_overhead_min_sec": 0.0,
            "communication_overhead_max_sec": 0.0,
            "communication_overhead_p50_sec": 0.0,
            "communication_overhead_p95_sec": 0.0,
            "communication_overhead_p99_sec": 0.0,
            "communication_overhead_ratio_avg": 0.0,
            "communication_overhead_ratio_total": 0.0,
        }

    latency_total = float(np.sum(latency))
    return {
        "communication_overhead_total_sec": float(np.sum(overhead)),
        "communication_overhead_avg_sec": float(np.mean(overhead)),
        "communication_overhead_std_sec": float(np.std(overhead)),
        "communication_overhead_min_sec": float(np.min(overhead)),
        "communication_overhead_max_sec": float(np.max(overhead)),
        "communication_overhead_p50_sec": float(np.percentile(overhead, 50)),
        "communication_overhead_p95_sec": float(np.percentile(overhead, 95)),
        "communication_overhead_p99_sec": float(np.percentile(overhead, 99)),
        "communication_overhead_ratio_avg": float(np.mean(ratios)),
        "communication_overhead_ratio_total": (
            float(np.sum(overhead) / latency_total) if latency_total > 0.0 else 0.0
        ),
    }


def infer_topology_label(row: dict[str, Any]) -> str:
    experiment_id = str(row.get("experiment_id", ""))
    mapping = {
        "exp1_1": "Single Node Baseline",
        "exp1_2": "Single Node Early Exit",
        "exp1_3": "Homogeneous 2 Workers",
        "exp1_4": "Homogeneous 3 Workers",
        "exp1_5": "Heterogeneous Pi + Jetson",
        "exp1_6": "Heterogeneous 2 Pis + Jetson",
        "exp2": "Homogeneous Multi-Model",
    }
    return mapping.get(experiment_id, experiment_id or str(row.get("system_name", "Unknown")))


def infer_category(row: dict[str, Any]) -> str:
    experiment_id = str(row.get("experiment_id", ""))
    if experiment_id == "exp1_1":
        return "baseline"
    if experiment_id == "exp1_2":
        return "single_node_ee"
    if experiment_id in {"exp1_3", "exp1_4"}:
        return "distributed_homogeneous"
    if experiment_id in {"exp1_5", "exp1_6"}:
        return "distributed_heterogeneous"
    if experiment_id == "exp2":
        return "distributed_multi_model"
    return "other"


def extract_worker_ids(row: dict[str, Any]) -> list[str]:
    worker_ids: set[str] = set()
    for key in row:
        if not key.endswith("_compute_time_total_sec"):
            continue
        if key.startswith("remote_"):
            continue
        if key.startswith("model_"):
            continue
        worker_ids.add(key[: -len("_compute_time_total_sec")])
    return sorted(worker_ids)


def load_summary_dataframe(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in load_metrics_files(results_dir):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            row = json.load(handle)

        if "communication_overhead_avg_sec" not in row:
            row.update(_compute_communication_overhead_fields(metrics_path))

        row["metrics_path"] = str(metrics_path.resolve())
        row["topology_label"] = infer_topology_label(row)
        row["category"] = infer_category(row)
        row["worker_ids"] = ",".join(extract_worker_ids(row))
        row["num_workers"] = len(extract_worker_ids(row))
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")

    df = pd.DataFrame(rows)
    if "experiment_id" in df.columns:
        df = df.sort_values("experiment_id").reset_index(drop=True)
    return df


def write_summary_bundle(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "combined_metrics.csv"
    json_path = output_dir / "combined_metrics.json"
    markdown_path = output_dir / "experiment_overview.md"

    export_df = df.copy()
    export_df.to_csv(csv_path, index=False)
    export_df.to_json(json_path, orient="records", indent=2)

    lines = [
        "# Experiment 1 Overview",
        "",
        "Combined metrics dataset generated from all available `metrics.json` files.",
        "",
        "## Included Runs",
        "",
    ]

    for _, row in df.iterrows():
        lines.append(
            "- {exp}: {label} | accuracy={acc:.2f}% | throughput={thr:.3f} samples/s | avg_latency={lat:.3f}s".format(
                exp=row.get("experiment_id", "unknown"),
                label=row.get("topology_label", "unknown"),
                acc=float(row.get("accuracy", 0.0)),
                thr=float(row.get("throughput_samples_per_sec", 0.0)),
                lat=float(row.get("avg_latency_sec", 0.0)),
            )
        )

    best_throughput = df.loc[df["throughput_samples_per_sec"].idxmax()]
    lowest_energy = df.loc[df["energy_kWh"].fillna(df.get("system_energy_kWh_total")).idxmin()] if "energy_kWh" in df.columns else None

    lines.extend(
        [
            "",
            "## Highlights",
            "",
            "- Highest throughput: {exp} ({label}) at {value:.3f} samples/s".format(
                exp=best_throughput["experiment_id"],
                label=best_throughput["topology_label"],
                value=float(best_throughput["throughput_samples_per_sec"]),
            ),
        ]
    )

    if lowest_energy is not None:
        energy_value = lowest_energy.get("energy_kWh")
        if pd.isna(energy_value):
            energy_value = lowest_energy.get("system_energy_kWh_total")
        lines.append(
            "- Lowest total energy figure: {exp} ({label}) at {value:.6f} kWh".format(
                exp=lowest_energy["experiment_id"],
                label=lowest_energy["topology_label"],
                value=float(energy_value),
            )
        )

    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    df = load_summary_dataframe(results_dir)
    write_summary_bundle(df, output_dir)


if __name__ == "__main__":
    main()
