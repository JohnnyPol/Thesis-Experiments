from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.summary import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    extract_worker_ids,
    load_summary_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis plots from Experiment 1 metrics."
    )
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _prepare_plot_dir(output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_performance_overview(df: pd.DataFrame, plot_dir: Path) -> None:
    labels = df["topology_label"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, df["accuracy"], color="#4C78A8")
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Percent")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")

    axes[1].bar(x, df["throughput_samples_per_sec"], color="#72B7B2")
    axes[1].set_title("Throughput")
    axes[1].set_ylabel("Samples / sec")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")

    axes[2].bar(x, df["avg_latency_sec"], color="#F58518")
    axes[2].set_title("Average Latency")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")

    _save(fig, plot_dir / "performance_overview.png")


def plot_energy_emissions(df: pd.DataFrame, plot_dir: Path) -> None:
    labels = df["topology_label"].tolist()
    x = np.arange(len(labels))

    energy_series = df["system_energy_kWh_total"] if "system_energy_kWh_total" in df.columns else df["energy_kWh"]
    carbon_series = df["system_carbon_kg_total"] if "system_carbon_kg_total" in df.columns else df["carbon_kg"]
    energy_series = energy_series.fillna(df.get("energy_kWh"))
    carbon_series = carbon_series.fillna(df.get("carbon_kg"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x, energy_series, color="#54A24B")
    axes[0].set_title("Energy Consumption")
    axes[0].set_ylabel("kWh")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")

    axes[1].bar(x, carbon_series, color="#E45756")
    axes[1].set_title("Carbon Emissions")
    axes[1].set_ylabel("kg CO2eq")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")

    _save(fig, plot_dir / "energy_emissions.png")


def plot_network_protocol(df: pd.DataFrame, plot_dir: Path) -> None:
    distributed_df = df[df["mode"].astype(str).str.startswith("distributed")].copy()
    if distributed_df.empty:
        return

    labels = distributed_df["topology_label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width / 2,
        distributed_df["protocol_bytes_total"] / 1e9,
        width=width,
        label="Protocol GB",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        distributed_df["master_network_total_bytes"] / 1e9,
        width=width,
        label="Master Network GB",
        color="#B279A2",
    )
    ax.set_title("Distributed Communication Overhead")
    ax.set_ylabel("GB")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend()

    _save(fig, plot_dir / "distributed_network_protocol.png")


def plot_exit_distribution(df: pd.DataFrame, plot_dir: Path) -> None:
    ee_df = df[df["experiment_id"] != "exp1_1"].copy()
    if ee_df.empty:
        return

    labels = ee_df["topology_label"].tolist()
    x = np.arange(len(labels))
    width = 0.6

    exit_keys = ["exit_0_ratio", "exit_1_ratio", "exit_2_ratio", "exit_3_ratio"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    bottom = np.zeros(len(ee_df))

    fig, ax = plt.subplots(figsize=(12, 6))
    for exit_key, color in zip(exit_keys, colors):
        values = ee_df[exit_key].fillna(0.0).to_numpy()
        ax.bar(x, values, width=width, bottom=bottom, label=exit_key, color=color)
        bottom += values

    ax.set_title("Exit Distribution Across Early-Exit Experiments")
    ax.set_ylabel("Ratio")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend()

    _save(fig, plot_dir / "exit_distribution.png")


def plot_worker_compute_breakdown(df: pd.DataFrame, plot_dir: Path) -> None:
    distributed_df = df[df["mode"].astype(str).str.startswith("distributed")].copy()
    if distributed_df.empty:
        return

    experiment_labels = distributed_df["topology_label"].tolist()
    worker_sets = [extract_worker_ids(row.to_dict()) for _, row in distributed_df.iterrows()]
    worker_ids = sorted({worker_id for worker_list in worker_sets for worker_id in worker_list})

    x = np.arange(len(distributed_df))
    width = 0.8 / max(len(worker_ids), 1)

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, worker_id in enumerate(worker_ids):
        values = distributed_df[f"{worker_id}_compute_time_total_sec"] if f"{worker_id}_compute_time_total_sec" in distributed_df.columns else pd.Series([0.0] * len(distributed_df))
        ax.bar(
            x + (idx - (len(worker_ids) - 1) / 2) * width,
            values.fillna(0.0),
            width=width,
            label=worker_id,
        )

    ax.set_title("Worker Compute Time Breakdown")
    ax.set_ylabel("Total Compute Time (sec)")
    ax.set_xticks(x, experiment_labels, rotation=25, ha="right")
    ax.legend()

    _save(fig, plot_dir / "worker_compute_breakdown.png")


def main() -> None:
    args = parse_args()
    df = load_summary_dataframe(Path(args.results_dir))
    plot_dir = _prepare_plot_dir(Path(args.output_dir))

    plot_performance_overview(df, plot_dir)
    plot_energy_emissions(df, plot_dir)
    plot_network_protocol(df, plot_dir)
    plot_exit_distribution(df, plot_dir)
    plot_worker_compute_breakdown(df, plot_dir)


if __name__ == "__main__":
    main()
