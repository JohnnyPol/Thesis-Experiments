from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_LABELS = {
    "exp1_1": "1.1",
    "exp1_2": "1.2",
    "exp1_3": "1.3",
    "exp2": "2.0",
    "exp3_1": "3.1",
    "exp3_2": "3.2",
    "exp3_3": "3.3",
}

EXPERIMENT_SORT_ORDER = {
    "1.1": 0,
    "1.2": 1,
    "1.3": 2,
    "2.0": 3,
    "3.1": 4,
    "3.2": 5,
    "3.3": 6,
}

MODEL_SORT_ORDER = {
    "resnet18": 0,
    "resnet34": 1,
}

WORKER_IDS = ("worker1", "worker2", "worker3")
EXIT_IDS = (0, 1, 2, 3)
DEFAULT_FIGURE_FORMATS = ("png", "pdf")

THESIS_VISUALIZATIONS_DIR = Path("results/thesis_visualizations")
EXP1_OUTPUT_SUBDIR = "exp1_single_model"
EXP2_EXP3_OUTPUT_SUBDIR = "exp2_exp3"
ENERGY_OUTPUT_SUBDIR = "energy"
CIFAR10_DATASET_NAME = "cifar10"
DATASET_NAME_ALIASES = {
    "cifar10.1": "cifar10_1",
    "cifar10-1": "cifar10_1",
}

EXP2_RESULTS_DIR = Path("results/exp2_multi_model")
EXP3_RESULTS_DIR = Path("results/exp3_memory_aware_multi_model")
ALL_RESULTS_DIRS = (
    Path("results/exp1_single_model"),
    EXP2_RESULTS_DIR,
    EXP3_RESULTS_DIR,
)
ALL_DISTRIBUTED_RESULTS_DIRS = (
    Path("results/exp1_single_model"),
    EXP2_RESULTS_DIR,
    EXP3_RESULTS_DIR,
)

PLACEMENT_LABELS = {
    "1.1": "baseline",
    "1.2": "single-node",
    "1.3": "distributed",
    "2.0": "naive",
    "3.1": "first-split",
    "3.2": "staggered-late",
    "3.3": "cross-balanced",
}

MODEL_COLORS = {
    "ResNet18": "#4C78A8",
    "ResNet34": "#F58518",
    "ResNet18 Baseline": "#72B7B2",
    "ResNet34 Baseline": "#E45756",
    "ResNet18EE": "#4C78A8",
    "ResNet34EE": "#F58518",
}

EXIT_COLORS = {
    "Exit 0": "#4C78A8",
    "Exit 1": "#F58518",
    "Exit 2": "#54A24B",
    "Exit 3": "#E45756",
}

WORKER_COLORS = {
    "worker1": "#4C78A8",
    "worker2": "#F58518",
    "worker3": "#54A24B",
}

ENERGY_COMPONENT_COLORS = {
    "master": "#7F7F7F",
    "worker1": "#4C78A8",
    "worker2": "#F58518",
    "worker3": "#54A24B",
}


def build_exp1_3_worker_utilization_plot(
    results_dir: str | Path = "results/exp1_single_model",
    output_dir: str | Path = "results/thesis_visualizations/exp1_single_model",
    dataset_name: str | None = "cifar10",
    figure_name: str = "exp1_3_worker_utilization",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate a grouped barplot of worker utilization for Experiment 1.3.

    The plot compares ResNet18EE and ResNet34EE across worker1, worker2, and
    worker3. Utilization is shown as a percentage.
    """
    exp13_metrics = [
        metrics
        for _, metrics in _load_latest_experiment_metrics(
            Path(results_dir),
            dataset_name=dataset_name,
        )
        if str(metrics.get("experiment_id", "")).startswith("exp1_3")
    ]

    rows: list[dict[str, Any]] = []
    for metrics in exp13_metrics:
        model = _model_label(str(metrics.get("model_name", "")))
        for worker_id in WORKER_IDS:
            utilization = _optional_float(metrics.get(f"{worker_id}_node_utilization"))
            rows.append(
                {
                    "experiment": _experiment_label(
                        str(metrics.get("experiment_id", ""))
                    ),
                    "model": model,
                    "worker": worker_id,
                    "node_utilization": (
                        utilization * 100.0 if utilization is not None else 0.0
                    ),
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No Experiment 1.3 metrics found under {Path(results_dir)}"
            + (f" for dataset '{dataset_name}'" if dataset_name else "")
        )

    data = pd.DataFrame(rows)
    data["_model_sort"] = data["model"].map(_model_sort_value)
    data["_worker_sort"] = data["worker"].map(
        {worker: i for i, worker in enumerate(WORKER_IDS)}
    )
    data = data.sort_values(["_worker_sort", "_model_sort"], kind="stable")
    data = data.drop(columns=["_model_sort", "_worker_sort"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    workers = list(WORKER_IDS)
    models = sorted(data["model"].unique(), key=_model_sort_value)
    x = np.arange(len(workers))
    width = 0.34

    for index, model in enumerate(models):
        model_data = data[data["model"] == model].set_index("worker")
        values = [
            float(model_data.loc[worker, "node_utilization"]) for worker in workers
        ]
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model),
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in values],
            padding=3,
            fontsize=8,
        )

    ax.set_title("Experiment 1.3 Worker Utilization")
    ax.set_xlabel("Worker")
    ax.set_ylabel("Node utilization (%)")
    ax.set_xticks(x, workers)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data


def build_experiment1_exit_distribution_plot(
    results_dir: str | Path = "results/exp1_single_model",
    output_dir: str | Path = "results/thesis_visualizations/exp1_single_model",
    dataset_name: str | None = "cifar10",
    experiment_prefixes: Iterable[str] = ("exp1_2", "exp1_3"),
    figure_name: str = "experiment1_exit_distribution",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate a stacked barplot of early-exit distribution for Experiment 1.

    By default, the plot includes Experiment 1.2 and 1.3 early-exit runs. Exit
    ratios are shown as percentages.
    """
    prefixes = tuple(experiment_prefixes)
    rows: list[dict[str, Any]] = []

    for _, metrics in _load_latest_experiment_metrics(
        Path(results_dir),
        dataset_name=dataset_name,
    ):
        experiment_id = str(metrics.get("experiment_id", ""))
        if not experiment_id.startswith(prefixes):
            continue
        if metrics.get("exit_0_ratio") is None:
            continue

        row: dict[str, Any] = {
            "experiment": _experiment_label(experiment_id),
            "model": _model_label(str(metrics.get("model_name", ""))),
        }
        for exit_id in EXIT_IDS:
            row[f"exit{exit_id}"] = (
                _optional_float(metrics.get(f"exit_{exit_id}_ratio")) or 0.0
            ) * 100.0
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No early-exit Experiment 1 metrics found under {Path(results_dir)}"
            + (f" for dataset '{dataset_name}'" if dataset_name else "")
        )

    data = pd.DataFrame(rows)
    data["_experiment_sort"] = data["experiment"].map(EXPERIMENT_SORT_ORDER)
    data["_model_sort"] = data["model"].map(_model_sort_value)
    data = data.sort_values(
        ["_experiment_sort", "_model_sort"],
        kind="stable",
    ).drop(columns=["_experiment_sort", "_model_sort"])
    data = data.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = [f"{row.experiment}\n{row.model}" for row in data.itertuples()]
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    width = 0.64

    for exit_id in EXIT_IDS:
        column = f"exit{exit_id}"
        label = f"Exit {exit_id}"
        values = data[column].astype(float).to_numpy()
        bars = ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            label=label,
            color=EXIT_COLORS[label],
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 5.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        bottom += values

    ax.set_title("Experiment 1 Exit Distribution")
    ax.set_xlabel("Experiment and model")
    ax.set_ylabel("Samples (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data


def build_exp2_exp3_throughput_speedup_plot(
    exp2_results_dir: str | Path = EXP2_RESULTS_DIR,
    exp3_results_dir: str | Path = EXP3_RESULTS_DIR,
    output_dir: str | Path = "results/thesis_visualizations/exp2_exp3",
    dataset_name: str | None = "cifar10",
    figure_name: str = "exp2_exp3_throughput_speedup",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate a grouped barplot of Experiment 3 throughput speedup over Exp2.
    """
    data = _build_exp2_exp3_performance_data(
        exp2_results_dir=Path(exp2_results_dir),
        exp3_results_dir=Path(exp3_results_dir),
        dataset_name=dataset_name,
    )
    data = data[data["experiment"] != "2.0"].reset_index(drop=True)
    if data.empty:
        raise FileNotFoundError("No Experiment 3 metrics found for speedup plot.")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    experiments = _ordered_unique(data["experiment"])
    models = _ordered_models(data["model"])
    x = np.arange(len(experiments))
    width = 0.34

    for index, model in enumerate(models):
        model_data = data[data["model"] == model].set_index("experiment")
        values = [
            float(model_data.loc[experiment, "speedup_vs_exp2"])
            for experiment in experiments
        ]
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model),
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}x" for value in values],
            padding=3,
            fontsize=8,
        )

    ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_title("Experiment 3 Throughput Speedup over Experiment 2")
    ax.set_xlabel("Experiment 3 placement")
    ax.set_ylabel("Throughput speedup")
    ax.set_xticks(x, [f"{experiment}\n{_placement_label(experiment)}" for experiment in experiments])
    ax.set_ylim(0, max(2.0, float(data["speedup_vs_exp2"].max()) * 1.18))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data[
        [
            "experiment",
            "placement",
            "model",
            "throughput",
            "speedup_vs_exp2",
            "inference_time_reduction_pct",
        ]
    ]


def build_exp2_best_exp3_worker_utilization_plot(
    exp2_results_dir: str | Path = EXP2_RESULTS_DIR,
    exp3_results_dir: str | Path = EXP3_RESULTS_DIR,
    output_dir: str | Path = "results/thesis_visualizations/exp2_exp3",
    dataset_name: str | None = "cifar10",
    best_experiment: str = "3.2",
    figure_name: str = "exp2_best_exp3_worker_utilization",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate worker-utilization barplots for Exp2 vs the selected best Exp3.
    """
    data = _build_distributed_node_utilization_data(
        [Path(exp2_results_dir), Path(exp3_results_dir)],
        dataset_name=dataset_name,
    )
    data = data[data["experiment"].isin(["2.0", best_experiment])].reset_index(
        drop=True
    )
    if data.empty:
        raise FileNotFoundError("No Exp2/best-Exp3 utilization metrics found.")

    models = _ordered_models(data["model"])
    fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    workers = list(WORKER_IDS)
    experiments = ["2.0", best_experiment]
    width = 0.34
    x = np.arange(len(workers))

    for ax, model in zip(axes, models):
        model_data = data[data["model"] == model]
        for index, experiment in enumerate(experiments):
            experiment_data = model_data[model_data["experiment"] == experiment].set_index(
                "worker"
            )
            values = [
                float(experiment_data.loc[worker, "node_utilization"])
                for worker in workers
            ]
            offset = (index - (len(experiments) - 1) / 2) * width
            label = f"{experiment} {_placement_label(experiment)}"
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=label,
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.1f}%" for value in values],
                padding=3,
                fontsize=8,
            )

        ax.set_title(model)
        ax.set_xlabel("Worker")
        ax.set_xticks(x, workers)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Node utilization (%)")
    axes[0].set_ylim(0, 110)
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Experiment 2 vs Experiment {best_experiment} Worker Utilization")

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data


def build_exp2_exp3_communication_overhead_plot(
    exp2_results_dir: str | Path = EXP2_RESULTS_DIR,
    exp3_results_dir: str | Path = EXP3_RESULTS_DIR,
    output_dir: str | Path = "results/thesis_visualizations/exp2_exp3",
    dataset_name: str | None = "cifar10",
    figure_name: str = "exp2_exp3_communication_overhead",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate a grouped barplot of communication overhead ratio.
    """
    data = _build_exp2_exp3_performance_data(
        exp2_results_dir=Path(exp2_results_dir),
        exp3_results_dir=Path(exp3_results_dir),
        dataset_name=dataset_name,
    )
    if data.empty:
        raise FileNotFoundError("No Exp2/Exp3 communication overhead metrics found.")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    experiments = _ordered_unique(data["experiment"])
    models = _ordered_models(data["model"])
    x = np.arange(len(experiments))
    width = 0.34

    for index, model in enumerate(models):
        model_data = data[data["model"] == model].set_index("experiment")
        values = [
            float(model_data.loc[experiment, "communication_overhead_ratio_pct"])
            for experiment in experiments
        ]
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model),
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in values],
            padding=3,
            fontsize=8,
        )

    ax.set_title("Communication Overhead Ratio")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Communication overhead (%)")
    ax.set_xticks(x, [f"{experiment}\n{_placement_label(experiment)}" for experiment in experiments])
    ax.set_ylim(0, max(65.0, float(data["communication_overhead_ratio_pct"].max()) * 1.16))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data[
        [
            "experiment",
            "placement",
            "model",
            "communication_overhead_avg",
            "communication_overhead_ratio_pct",
            "communication_overhead_reduction_pct",
        ]
    ]


def build_distributed_node_utilization_plot(
    results_dirs: Iterable[str | Path] = ALL_DISTRIBUTED_RESULTS_DIRS,
    output_dir: str | Path = "results/thesis_visualizations/distributed",
    dataset_name: str | None = "cifar10",
    figure_name: str = "distributed_node_utilization",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate node-utilization barplots for all distributed experiments.
    """
    data = _build_distributed_node_utilization_data(
        [Path(results_dir) for results_dir in results_dirs],
        dataset_name=dataset_name,
    )
    if data.empty:
        raise FileNotFoundError("No distributed utilization metrics found.")

    models = _ordered_models(data["model"])
    experiments = _ordered_unique(data["experiment"])
    fig, axes = plt.subplots(1, len(models), figsize=(7.2 * len(models), 4.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    x = np.arange(len(experiments))
    width = 0.24
    for ax, model in zip(axes, models):
        model_data = data[data["model"] == model]
        for index, worker in enumerate(WORKER_IDS):
            worker_data = model_data[model_data["worker"] == worker].set_index(
                "experiment"
            )
            values = [
                float(worker_data.loc[experiment, "node_utilization"])
                if experiment in worker_data.index
                else 0.0
                for experiment in experiments
            ]
            offset = (index - (len(WORKER_IDS) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=worker,
                color=WORKER_COLORS.get(worker),
            )

        ax.set_title(model)
        ax.set_xlabel("Experiment")
        ax.set_xticks(
            x,
            [f"{experiment}\n{_placement_label(experiment)}" for experiment in experiments],
            rotation=0,
        )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Node utilization (%)")
    axes[0].set_ylim(0, 110)
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Node Utilization Across Distributed Experiments")

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data


def build_energy_per_sample_plot(
    results_dirs: Iterable[str | Path] = ALL_RESULTS_DIRS,
    output_dir: str | Path = "results/thesis_visualizations/energy",
    dataset_name: str | None = "cifar10",
    figure_name: str = "energy_per_sample",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate a grouped barplot of system energy per sample.
    """
    data = _build_energy_data(
        [Path(results_dir) for results_dir in results_dirs],
        dataset_name=dataset_name,
    )
    if data.empty:
        raise FileNotFoundError("No energy metrics found.")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    experiments = _ordered_unique(data["experiment"])
    architectures = _ordered_architectures(data["architecture"])
    x = np.arange(len(experiments))
    width = min(0.34, 0.78 / max(len(architectures), 1))

    for index, architecture in enumerate(architectures):
        model_data = data[data["architecture"] == architecture].set_index(
            "experiment"
        )
        values = [
            (
                float(model_data.loc[experiment, "energy_per_sample_J"])
                if experiment in model_data.index
                else np.nan
            )
            for experiment in experiments
        ]
        offset = (index - (len(architectures) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=architecture,
            color=MODEL_COLORS.get(architecture),
        )
        ax.bar_label(
            bars,
            labels=[
                f"{value:.2f}" if not np.isnan(value) else "" for value in values
            ],
            padding=3,
            fontsize=8,
        )

    ax.set_title("System Energy per Sample")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Energy per sample (J)")
    ax.set_xticks(
        x,
        [f"{experiment}\n{_placement_label(experiment)}" for experiment in experiments],
    )
    ax.set_ylim(0, max(1.0, float(data["energy_per_sample_J"].max()) * 1.18))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data[
        [
            "experiment",
            "placement",
            "architecture",
            "model",
            "num_samples",
            "system_energy_kWh",
            "energy_per_sample_J",
        ]
    ]


def build_energy_breakdown_plot(
    results_dirs: Iterable[str | Path] = ALL_DISTRIBUTED_RESULTS_DIRS,
    output_dir: str | Path = "results/thesis_visualizations/energy",
    dataset_name: str | None = "cifar10",
    figure_name: str = "distributed_energy_breakdown",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate stacked energy-per-sample bars split by master and workers.
    """
    data = _build_energy_breakdown_data(
        [Path(results_dir) for results_dir in results_dirs],
        dataset_name=dataset_name,
    )
    if data.empty:
        raise FileNotFoundError("No distributed energy breakdown metrics found.")

    architectures = _ordered_architectures(data["architecture"])
    experiments = _ordered_unique(data["experiment"])
    components = ["master", *WORKER_IDS]
    fig, axes = plt.subplots(
        1,
        len(architectures),
        figsize=(7.2 * len(architectures), 4.8),
        sharey=True,
    )
    if len(architectures) == 1:
        axes = [axes]

    x = np.arange(len(experiments))
    for ax, architecture in zip(axes, architectures):
        model_data = data[data["architecture"] == architecture]
        bottom = np.zeros(len(experiments))

        for component in components:
            component_data = model_data[
                model_data["component"] == component
            ].set_index("experiment")
            values = np.array(
                [
                    (
                        float(component_data.loc[experiment, "energy_per_sample_J"])
                        if experiment in component_data.index
                        else 0.0
                    )
                    for experiment in experiments
                ]
            )
            ax.bar(
                x,
                values,
                bottom=bottom,
                label=component,
                color=ENERGY_COMPONENT_COLORS.get(component),
            )
            bottom += values

        for position, total in zip(x, bottom):
            ax.text(
                position,
                total + 0.04,
                f"{total:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(architecture)
        ax.set_xlabel("Experiment")
        ax.set_xticks(
            x,
            [f"{experiment}\n{_placement_label(experiment)}" for experiment in experiments],
        )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Energy per sample (J)")
    axes[0].set_ylim(0, max(1.0, float(data.groupby(["architecture", "experiment"])["energy_per_sample_J"].sum().max()) * 1.18))
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Distributed Energy per Sample Breakdown")

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data


def build_energy_delay_product_plot(
    results_dirs: Iterable[str | Path] = ALL_RESULTS_DIRS,
    output_dir: str | Path = "results/thesis_visualizations/energy",
    dataset_name: str | None = "cifar10",
    figure_name: str = "energy_delay_product",
    formats: Iterable[str] = DEFAULT_FIGURE_FORMATS,
) -> pd.DataFrame:
    """
    Generate a grouped barplot of per-sample Energy-Delay Product.
    """
    data = _build_energy_data(
        [Path(results_dir) for results_dir in results_dirs],
        dataset_name=dataset_name,
    )
    data = data.dropna(subset=["energy_delay_product_Js"]).reset_index(drop=True)
    if data.empty:
        raise FileNotFoundError("No energy-delay product metrics found.")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    experiments = _ordered_unique(data["experiment"])
    architectures = _ordered_architectures(data["architecture"])
    x = np.arange(len(experiments))
    width = min(0.34, 0.78 / max(len(architectures), 1))

    for index, architecture in enumerate(architectures):
        model_data = data[data["architecture"] == architecture].set_index(
            "experiment"
        )
        values = [
            (
                float(model_data.loc[experiment, "energy_delay_product_Js"])
                if experiment in model_data.index
                else np.nan
            )
            for experiment in experiments
        ]
        offset = (index - (len(architectures) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=architecture,
            color=MODEL_COLORS.get(architecture),
        )
        ax.bar_label(
            bars,
            labels=[
                f"{value:.2f}" if not np.isnan(value) else "" for value in values
            ],
            padding=3,
            fontsize=8,
        )

    ax.set_title("Energy-Delay Product per Sample")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Energy-Delay Product (J s)")
    ax.set_xticks(
        x,
        [f"{experiment}\n{_placement_label(experiment)}" for experiment in experiments],
    )
    ax.set_ylim(0, max(1.0, float(data["energy_delay_product_Js"].max()) * 1.18))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    _save_figure(
        fig,
        Path(output_dir),
        _artifact_name(figure_name, dataset_name),
        formats,
    )
    return data[
        [
            "experiment",
            "placement",
            "architecture",
            "model",
            "energy_per_sample_J",
            "inference_time",
            "energy_delay_product_Js",
        ]
    ]


def _build_exp2_exp3_performance_data(
    exp2_results_dir: Path,
    exp3_results_dir: Path,
    dataset_name: str | None,
) -> pd.DataFrame:
    raw_rows = []
    for metrics_path, metrics in _load_latest_metrics(
        [exp2_results_dir, exp3_results_dir],
        dataset_name=dataset_name,
    ):
        experiment = _experiment_label(str(metrics.get("experiment_id", "")))
        if experiment not in {"2.0", "3.1", "3.2", "3.3"}:
            continue
        raw_rows.append(_build_performance_row(metrics, metrics_path))

    if not raw_rows:
        return pd.DataFrame()

    baseline_by_model = {
        row["model"]: row for row in raw_rows if row["experiment"] == "2.0"
    }
    rows = []
    for row in raw_rows:
        row = dict(row)
        baseline = baseline_by_model.get(row["model"])
        if baseline is None:
            row["speedup_vs_exp2"] = np.nan
            row["inference_time_reduction_pct"] = np.nan
            row["communication_overhead_reduction_pct"] = np.nan
        else:
            row["speedup_vs_exp2"] = _safe_ratio(
                row["throughput"],
                baseline["throughput"],
            )
            row["inference_time_reduction_pct"] = _reduction_percent(
                baseline["inference_time"],
                row["inference_time"],
            )
            row["communication_overhead_reduction_pct"] = _reduction_percent(
                baseline["communication_overhead_avg"],
                row["communication_overhead_avg"],
            )
        rows.append(row)

    data = pd.DataFrame(rows)
    data["_experiment_sort"] = data["experiment"].map(EXPERIMENT_SORT_ORDER)
    data["_model_sort"] = data["model"].map(_model_sort_value)
    data = data.sort_values(["_model_sort", "_experiment_sort"], kind="stable")
    return data.drop(columns=["_experiment_sort", "_model_sort"]).reset_index(
        drop=True
    )


def _build_performance_row(
    metrics: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any]:
    experiment_id = str(metrics.get("experiment_id", ""))
    experiment = _experiment_label(experiment_id)
    return {
        "experiment": experiment,
        "placement": _placement_label(experiment),
        "model": _model_label(str(metrics.get("model_name", ""))),
        "throughput": _optional_float(metrics.get("throughput_samples_per_sec")),
        "inference_time": _resolve_avg_inference_time(metrics, metrics_path),
        "communication_overhead_avg": _optional_float(
            metrics.get("communication_overhead_avg_sec")
        ),
        "communication_overhead_ratio_pct": (
            (_optional_float(metrics.get("communication_overhead_ratio_total")) or 0.0)
            * 100.0
        ),
    }


def _build_distributed_node_utilization_data(
    results_dirs: Iterable[Path],
    dataset_name: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, metrics in _load_latest_metrics(results_dirs, dataset_name=dataset_name):
        experiment = _experiment_label(str(metrics.get("experiment_id", "")))
        if experiment not in {"1.3", "2.0", "3.1", "3.2", "3.3"}:
            continue
        model = _model_label(str(metrics.get("model_name", "")))
        for worker in WORKER_IDS:
            utilization = _optional_float(metrics.get(f"{worker}_node_utilization"))
            if utilization is None:
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "placement": _placement_label(experiment),
                    "model": model,
                    "worker": worker,
                    "node_utilization": utilization * 100.0,
                }
            )

    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame(rows)
    data["_experiment_sort"] = data["experiment"].map(EXPERIMENT_SORT_ORDER)
    data["_model_sort"] = data["model"].map(_model_sort_value)
    data["_worker_sort"] = data["worker"].map(
        {worker: i for i, worker in enumerate(WORKER_IDS)}
    )
    data = data.sort_values(
        ["_model_sort", "_experiment_sort", "_worker_sort"],
        kind="stable",
    )
    return data.drop(
        columns=["_experiment_sort", "_model_sort", "_worker_sort"]
    ).reset_index(drop=True)


def _build_energy_data(
    results_dirs: Iterable[Path],
    dataset_name: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path, metrics in _load_latest_metrics(
        results_dirs,
        dataset_name=dataset_name,
    ):
        row = _build_energy_row(metrics, metrics_path)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame(rows)
    data["_experiment_sort"] = data["experiment"].map(EXPERIMENT_SORT_ORDER)
    data["_model_sort"] = data["architecture"].map(_model_sort_value)
    data = data.sort_values(
        ["_model_sort", "_experiment_sort"],
        kind="stable",
    )
    return data.drop(columns=["_experiment_sort", "_model_sort"]).reset_index(
        drop=True
    )


def _build_energy_row(
    metrics: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any] | None:
    experiment_id = str(metrics.get("experiment_id", ""))
    experiment = _experiment_label(experiment_id)
    total_energy_kwh = _resolve_total_energy_kwh(metrics)
    num_samples = _optional_float(metrics.get("num_samples"))

    if total_energy_kwh is None or num_samples is None or num_samples == 0.0:
        return None

    model_name = str(metrics.get("model_name", ""))
    energy_per_sample_j = total_energy_kwh * 3_600_000.0 / num_samples
    inference_time = _resolve_energy_inference_time(metrics, metrics_path, num_samples)
    total_carbon_kg = _resolve_total_carbon_kg(metrics)

    return {
        "experiment": experiment,
        "placement": _placement_label(experiment),
        "architecture": _model_architecture(model_name),
        "model": _model_label(model_name, experiment_id),
        "num_samples": int(num_samples),
        "system_energy_kWh": total_energy_kwh,
        "system_carbon_kg": total_carbon_kg,
        "energy_per_sample_J": energy_per_sample_j,
        "carbon_per_sample_g": (
            total_carbon_kg * 1000.0 / num_samples
            if total_carbon_kg is not None
            else np.nan
        ),
        "inference_time": inference_time,
        "energy_delay_product_Js": (
            energy_per_sample_j * inference_time
            if inference_time is not None
            else np.nan
        ),
    }


def _build_energy_breakdown_data(
    results_dirs: Iterable[Path],
    dataset_name: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, metrics in _load_latest_metrics(results_dirs, dataset_name=dataset_name):
        experiment_id = str(metrics.get("experiment_id", ""))
        experiment = _experiment_label(experiment_id)
        if experiment not in {"1.3", "2.0", "3.1", "3.2", "3.3"}:
            continue

        num_samples = _optional_float(metrics.get("num_samples"))
        if num_samples is None or num_samples == 0.0:
            continue

        model_name = str(metrics.get("model_name", ""))
        for component, key in [
            ("master", "master_energy_kWh"),
            *[(worker, f"{worker}_energy_kWh") for worker in WORKER_IDS],
        ]:
            energy_kwh = _optional_float(metrics.get(key))
            if energy_kwh is None:
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "placement": _placement_label(experiment),
                    "architecture": _model_architecture(model_name),
                    "model": _model_label(model_name, experiment_id),
                    "component": component,
                    "energy_kWh": energy_kwh,
                    "energy_per_sample_J": energy_kwh * 3_600_000.0 / num_samples,
                }
            )

    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame(rows)
    data["_experiment_sort"] = data["experiment"].map(EXPERIMENT_SORT_ORDER)
    data["_model_sort"] = data["architecture"].map(_model_sort_value)
    data["_component_sort"] = data["component"].map(
        {component: index for index, component in enumerate(["master", *WORKER_IDS])}
    )
    data = data.sort_values(
        ["_model_sort", "_experiment_sort", "_component_sort"],
        kind="stable",
    )
    return data.drop(
        columns=["_experiment_sort", "_model_sort", "_component_sort"]
    ).reset_index(drop=True)


def _normalize_dataset_name(dataset_name: str | None) -> str | None:
    if dataset_name is None:
        return None
    normalized = dataset_name.strip().lower()
    if not normalized:
        return None
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def _dataset_suffix(dataset_name: str | None) -> str:
    normalized = _normalize_dataset_name(dataset_name)
    if normalized is None or normalized == CIFAR10_DATASET_NAME:
        return ""
    return f"_{normalized}"


def _artifact_name(base_name: str, dataset_name: str | None) -> str:
    return f"{base_name}{_dataset_suffix(dataset_name)}"


def _dataset_output_dir(subdir: str, dataset_name: str | None) -> Path:
    normalized = _normalize_dataset_name(dataset_name)
    if normalized is None or normalized == CIFAR10_DATASET_NAME:
        return THESIS_VISUALIZATIONS_DIR / subdir
    return THESIS_VISUALIZATIONS_DIR / normalized / subdir


def _resolve_output_dir(
    requested_output_dir: str | Path | None,
    subdir: str,
    dataset_name: str | None,
) -> Path:
    if requested_output_dir is not None:
        return Path(requested_output_dir)
    return _dataset_output_dir(subdir, dataset_name)


def _load_latest_metrics(
    results_dirs: Iterable[Path],
    dataset_name: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    dataset_name = _normalize_dataset_name(dataset_name)
    latest_by_experiment: dict[str, tuple[Path, dict[str, Any]]] = {}

    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        for metrics_path in sorted(results_dir.rglob("metrics.json")):
            metrics = _read_json(metrics_path)
            experiment_id = str(metrics.get("experiment_id", ""))
            if not experiment_id:
                continue
            if dataset_name and str(metrics.get("dataset_name", "")) != dataset_name:
                continue
            existing = latest_by_experiment.get(experiment_id)
            if existing is None or metrics_path.parent.name > existing[0].parent.name:
                latest_by_experiment[experiment_id] = (metrics_path, metrics)

    return list(latest_by_experiment.values())


def _load_latest_experiment_metrics(
    results_dir: Path,
    dataset_name: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    dataset_name = _normalize_dataset_name(dataset_name)
    latest_by_experiment: dict[str, tuple[Path, dict[str, Any]]] = {}

    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        metrics = _read_json(metrics_path)
        experiment_id = str(metrics.get("experiment_id", ""))
        if not experiment_id:
            continue
        if dataset_name and str(metrics.get("dataset_name", "")) != dataset_name:
            continue

        existing = latest_by_experiment.get(experiment_id)
        if existing is None or metrics_path.parent.name > existing[0].parent.name:
            latest_by_experiment[experiment_id] = (metrics_path, metrics)

    return list(latest_by_experiment.values())


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _experiment_label(experiment_id: str) -> str:
    for prefix, label in EXPERIMENT_LABELS.items():
        if experiment_id.startswith(prefix):
            return label
    return experiment_id


def _placement_label(experiment: str) -> str:
    return PLACEMENT_LABELS.get(experiment, "-")


def _model_label(model_name: str, experiment_id: str = "") -> str:
    architecture = _model_architecture(model_name)
    if experiment_id.startswith("exp1_1") or "baseline" in model_name.lower():
        return f"{architecture} Baseline"
    return f"{architecture}EE" if architecture else model_name


def _model_architecture(model_name: str) -> str:
    normalized = model_name.lower()
    if "resnet34" in normalized:
        return "ResNet34"
    if "resnet18" in normalized:
        return "ResNet18"
    return ""


def _model_sort_value(model_label: str) -> int:
    normalized = model_label.lower()
    for key, value in MODEL_SORT_ORDER.items():
        if key in normalized:
            return value
    return len(MODEL_SORT_ORDER)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_total_energy_kwh(metrics: dict[str, Any]) -> float | None:
    for key in ("system_energy_kWh_total", "energy_kWh"):
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value

    master_energy = _optional_float(metrics.get("master_energy_kWh")) or 0.0
    worker_energy = _optional_float(metrics.get("workers_energy_kWh_total"))
    if worker_energy is not None:
        return master_energy + worker_energy

    worker_values = [
        _optional_float(metrics.get(f"{worker}_energy_kWh")) or 0.0
        for worker in WORKER_IDS
    ]
    total = master_energy + sum(worker_values)
    return total if total > 0.0 else None


def _resolve_total_carbon_kg(metrics: dict[str, Any]) -> float | None:
    for key in ("system_carbon_kg_total", "carbon_kg"):
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value

    master_carbon = _optional_float(metrics.get("master_carbon_kg")) or 0.0
    worker_carbon = _optional_float(metrics.get("workers_carbon_kg_total"))
    if worker_carbon is not None:
        return master_carbon + worker_carbon

    worker_values = [
        _optional_float(metrics.get(f"{worker}_carbon_kg")) or 0.0
        for worker in WORKER_IDS
    ]
    total = master_carbon + sum(worker_values)
    return total if total > 0.0 else None


def _resolve_energy_inference_time(
    metrics: dict[str, Any],
    metrics_path: Path,
    num_samples: float,
) -> float | None:
    inference_time = _resolve_avg_inference_time(metrics, metrics_path)
    if inference_time is not None:
        return inference_time

    total_inference_time = _optional_float(metrics.get("total_inference_time_sec"))
    if total_inference_time is None or num_samples == 0.0:
        return None
    return total_inference_time / num_samples


def _resolve_avg_inference_time(
    metrics: dict[str, Any],
    metrics_path: Path,
) -> float | None:
    value = metrics.get("avg_inference_time_sec")
    if value is not None:
        return _optional_float(value)

    value = metrics.get("avg_latency_sec")
    if value is not None:
        return _optional_float(value)

    per_sample_path = metrics_path.with_name("inference_times.csv")
    if not per_sample_path.exists():
        per_sample_path = metrics_path.with_name("latencies.csv")
    if not per_sample_path.exists():
        return None

    per_sample = pd.read_csv(per_sample_path)
    for column in ("inference_time_sec", "latency_sec"):
        if column in per_sample.columns:
            return float(per_sample[column].astype(float).mean())
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _reduction_percent(baseline: float | None, candidate: float | None) -> float:
    if baseline is None or candidate is None or baseline == 0.0:
        return float("nan")
    return (baseline - candidate) / baseline * 100.0


def _ordered_unique(values: pd.Series) -> list[str]:
    unique_values = list(dict.fromkeys(str(value) for value in values.tolist()))
    return sorted(unique_values, key=lambda value: EXPERIMENT_SORT_ORDER.get(value, 999))


def _ordered_models(values: pd.Series) -> list[str]:
    unique_values = list(dict.fromkeys(str(value) for value in values.tolist()))
    return sorted(unique_values, key=_model_sort_value)


def _ordered_architectures(values: pd.Series) -> list[str]:
    unique_values = list(dict.fromkeys(str(value) for value in values.tolist()))
    return sorted(unique_values, key=_model_sort_value)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    figure_name: str,
    formats: Iterable[str],
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()

    for figure_format in formats:
        fig.savefig(
            plots_dir / f"{figure_name}.{figure_format}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis plots from experiment metrics."
    )
    parser.add_argument(
        "--results-dir",
        default="results/exp1_single_model",
        help="Directory containing experiment run outputs.",
    )
    parser.add_argument(
        "--exp2-results-dir",
        default=str(EXP2_RESULTS_DIR),
        help="Directory containing Experiment 2 run outputs.",
    )
    parser.add_argument(
        "--exp3-results-dir",
        default=str(EXP3_RESULTS_DIR),
        help="Directory containing Experiment 3 run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where generated plot artifacts will be written. If omitted, "
            "a dataset-aware thesis_visualizations subdirectory is used."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="cifar10",
        help="Dataset filter. Use an empty string to include all datasets.",
    )
    parser.add_argument(
        "--plot",
        choices=(
            "all",
            "exp1_3_worker_utilization",
            "experiment1_exit_distribution",
            "exp2_exp3_throughput_speedup",
            "exp2_best_exp3_worker_utilization",
            "exp2_exp3_communication_overhead",
            "distributed_node_utilization",
            "energy_per_sample",
            "energy_breakdown",
            "energy_delay_product",
            "all_energy",
        ),
        default="all",
        help="Plot to generate.",
    )
    parser.add_argument(
        "--best-experiment",
        default="3.2",
        help="Best Experiment 3 label used for Exp2-vs-best utilization plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = _normalize_dataset_name(args.dataset_name)
    exp1_output_dir = _resolve_output_dir(
        args.output_dir,
        EXP1_OUTPUT_SUBDIR,
        dataset_name,
    )
    exp2_exp3_output_dir = _resolve_output_dir(
        args.output_dir,
        EXP2_EXP3_OUTPUT_SUBDIR,
        dataset_name,
    )
    energy_output_dir = _resolve_output_dir(
        args.output_dir,
        ENERGY_OUTPUT_SUBDIR,
        dataset_name,
    )
    generated = []

    if args.plot in {"all", "exp1_3_worker_utilization"}:
        try:
            data = build_exp1_3_worker_utilization_plot(
                results_dir=args.results_dir,
                output_dir=exp1_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_exp1_3_worker_utilization_plot: {exc}")
        else:
            generated.append("build_exp1_3_worker_utilization_plot")
            print("build_exp1_3_worker_utilization_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "experiment1_exit_distribution"}:
        try:
            data = build_experiment1_exit_distribution_plot(
                results_dir=args.results_dir,
                output_dir=exp1_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_experiment1_exit_distribution_plot: {exc}")
        else:
            generated.append("build_experiment1_exit_distribution_plot")
            print("build_experiment1_exit_distribution_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "exp2_exp3_throughput_speedup"}:
        try:
            data = build_exp2_exp3_throughput_speedup_plot(
                exp2_results_dir=args.exp2_results_dir,
                exp3_results_dir=args.exp3_results_dir,
                output_dir=exp2_exp3_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_exp2_exp3_throughput_speedup_plot: {exc}")
        else:
            generated.append("build_exp2_exp3_throughput_speedup_plot")
            print("build_exp2_exp3_throughput_speedup_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "exp2_best_exp3_worker_utilization"}:
        try:
            data = build_exp2_best_exp3_worker_utilization_plot(
                exp2_results_dir=args.exp2_results_dir,
                exp3_results_dir=args.exp3_results_dir,
                output_dir=exp2_exp3_output_dir,
                dataset_name=dataset_name,
                best_experiment=args.best_experiment,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_exp2_best_exp3_worker_utilization_plot: {exc}")
        else:
            generated.append("build_exp2_best_exp3_worker_utilization_plot")
            print("build_exp2_best_exp3_worker_utilization_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "exp2_exp3_communication_overhead"}:
        try:
            data = build_exp2_exp3_communication_overhead_plot(
                exp2_results_dir=args.exp2_results_dir,
                exp3_results_dir=args.exp3_results_dir,
                output_dir=exp2_exp3_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_exp2_exp3_communication_overhead_plot: {exc}")
        else:
            generated.append("build_exp2_exp3_communication_overhead_plot")
            print("build_exp2_exp3_communication_overhead_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "distributed_node_utilization"}:
        try:
            data = build_distributed_node_utilization_plot(
                results_dirs=ALL_DISTRIBUTED_RESULTS_DIRS,
                output_dir=exp2_exp3_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_distributed_node_utilization_plot: {exc}")
        else:
            generated.append("build_distributed_node_utilization_plot")
            print("build_distributed_node_utilization_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "all_energy", "energy_per_sample"}:
        try:
            data = build_energy_per_sample_plot(
                results_dirs=ALL_RESULTS_DIRS,
                output_dir=energy_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_energy_per_sample_plot: {exc}")
        else:
            generated.append("build_energy_per_sample_plot")
            print("build_energy_per_sample_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "all_energy", "energy_breakdown"}:
        try:
            data = build_energy_breakdown_plot(
                results_dirs=ALL_DISTRIBUTED_RESULTS_DIRS,
                output_dir=energy_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_energy_breakdown_plot: {exc}")
        else:
            generated.append("build_energy_breakdown_plot")
            print("build_energy_breakdown_plot:")
            print(data.to_string(index=False))

    if args.plot in {"all", "all_energy", "energy_delay_product"}:
        try:
            data = build_energy_delay_product_plot(
                results_dirs=ALL_RESULTS_DIRS,
                output_dir=energy_output_dir,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as exc:
            print(f"Skipped build_energy_delay_product_plot: {exc}")
        else:
            generated.append("build_energy_delay_product_plot")
            print("build_energy_delay_product_plot:")
            print(data.to_string(index=False))

    if generated:
        print("Generated plots: " + ", ".join(generated))
    else:
        print("No plots generated.")


if __name__ == "__main__":
    main()
