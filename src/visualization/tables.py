from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any

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

EXPERIMENT1_PREFIXES = ("exp1_1", "exp1_2", "exp1_3")

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

EXP2_RESULTS_DIR = Path("results/exp2_multi_model")
EXP3_RESULTS_DIR = Path("results/exp3_memory_aware_multi_model")
ALL_DISTRIBUTED_RESULTS_DIRS = (
    Path("results/exp1_single_model"),
    EXP2_RESULTS_DIR,
    EXP3_RESULTS_DIR,
)

PLACEMENT_LABELS = {
    "2.0": "naive",
    "3.1": "first-split",
    "3.2": "staggered-late",
    "3.3": "cross-balanced",
}


def build_experiment1_core_table(
    results_dir: str | Path = "results/exp1_single_model",
    output_dir: str | Path = "results/thesis_visualizations/exp1_single_model",
    dataset_name: str | None = "cifar10",
    table_name: str = "experiment1_core_metrics",
) -> pd.DataFrame:
    """
    Build the Experiment 1 comparison table and export it as CSV and LaTeX.

    The output columns are:
    experiment, accuracy, inference_time, throughput, model, exit0, exit1,
    exit2, exit3.

    Accuracy and exit columns are percentages. Inference time is seconds per
    sample or model-sample. Throughput is samples/sec.
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    tables_path = output_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path, metrics in _load_latest_experiment1_metrics(
        results_path,
        dataset_name=dataset_name,
    ):
        rows.append(_build_experiment1_row(metrics, metrics_path))

    if not rows:
        raise FileNotFoundError(
            f"No Experiment 1 metrics found under {results_path}"
            + (f" for dataset '{dataset_name}'" if dataset_name else "")
        )

    table = pd.DataFrame(rows)
    table = _sort_experiment1_table(table)
    table = _round_experiment1_table(table)

    csv_path = tables_path / f"{table_name}.csv"
    tex_path = tables_path / f"{table_name}.tex"

    table.to_csv(csv_path, index=False, na_rep="-")
    table.to_latex(
        tex_path,
        index=False,
        na_rep="-",
        escape=True,
        caption="Experiment 1 core metrics.",
        label="tab:experiment1-core-metrics",
    )

    return table


def build_exp2_exp3_comparison_table(
    exp2_results_dir: str | Path = EXP2_RESULTS_DIR,
    exp3_results_dir: str | Path = EXP3_RESULTS_DIR,
    output_dir: str | Path = "results/thesis_visualizations/exp2_exp3",
    dataset_name: str | None = "cifar10",
    table_name: str = "exp2_exp3_comparison",
    write_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Build the main Exp2/Exp3 comparison table.

    Speedup and reduction columns are computed against Experiment 2 for the
    matching architecture.
    """
    metrics_items = _load_latest_metrics(
        [Path(exp2_results_dir), Path(exp3_results_dir)],
        dataset_name=dataset_name,
    )
    rows = _build_exp2_exp3_rows(metrics_items)
    if not rows:
        raise FileNotFoundError("No Experiment 2 or Experiment 3 metrics found.")

    table = _sort_table(pd.DataFrame(rows))
    table = _round_table(
        table,
        {
            "accuracy": 2,
            "inference_time": 4,
            "throughput": 3,
            "speedup_vs_exp2": 3,
            "inference_time_reduction_pct": 2,
            "communication_overhead_avg": 4,
            "communication_overhead_ratio_pct": 2,
            "worker1_util": 2,
            "worker2_util": 2,
            "worker3_util": 2,
        },
    )
    if write_artifacts:
        _write_table(
            table,
            Path(output_dir),
            table_name,
            caption="Experiment 2 and Experiment 3 comparison.",
            label="tab:exp2-exp3-comparison",
        )
    return table


def build_exp2_exp3_speedup_table(
    exp2_results_dir: str | Path = EXP2_RESULTS_DIR,
    exp3_results_dir: str | Path = EXP3_RESULTS_DIR,
    output_dir: str | Path = "results/thesis_visualizations/exp2_exp3",
    dataset_name: str | None = "cifar10",
    table_name: str = "exp2_exp3_throughput_speedup",
) -> pd.DataFrame:
    """
    Build a plot-ready table of Experiment 3 throughput speedups over Exp2.
    """
    comparison = build_exp2_exp3_comparison_table(
        exp2_results_dir=exp2_results_dir,
        exp3_results_dir=exp3_results_dir,
        output_dir=output_dir,
        dataset_name=dataset_name,
        write_artifacts=False,
    )
    table = comparison[comparison["experiment"] != "2.0"][
        [
            "experiment",
            "placement",
            "model",
            "throughput",
            "speedup_vs_exp2",
            "inference_time_reduction_pct",
        ]
    ].copy()
    _write_table(
        table,
        Path(output_dir),
        table_name,
        caption="Experiment 3 throughput speedup over Experiment 2.",
        label="tab:exp2-exp3-throughput-speedup",
    )
    return table


def build_exp2_best_exp3_worker_utilization_table(
    exp2_results_dir: str | Path = EXP2_RESULTS_DIR,
    exp3_results_dir: str | Path = EXP3_RESULTS_DIR,
    output_dir: str | Path = "results/thesis_visualizations/exp2_exp3",
    dataset_name: str | None = "cifar10",
    best_experiment: str = "3.2",
    table_name: str = "exp2_best_exp3_worker_utilization",
) -> pd.DataFrame:
    """
    Build a worker-utilization table comparing naive Exp2 with the best Exp3.
    """
    table = build_distributed_node_utilization_table(
        results_dirs=[Path(exp2_results_dir), Path(exp3_results_dir)],
        output_dir=output_dir,
        dataset_name=dataset_name,
        table_name=table_name,
        write_artifacts=False,
    )
    table = table[table["experiment"].isin(["2.0", best_experiment])].reset_index(
        drop=True
    )
    _write_table(
        table,
        Path(output_dir),
        table_name,
        caption=f"Worker utilization for Experiment 2 and Experiment {best_experiment}.",
        label="tab:exp2-best-exp3-worker-utilization",
    )
    return table


def build_exp2_exp3_communication_overhead_table(
    exp2_results_dir: str | Path = EXP2_RESULTS_DIR,
    exp3_results_dir: str | Path = EXP3_RESULTS_DIR,
    output_dir: str | Path = "results/thesis_visualizations/exp2_exp3",
    dataset_name: str | None = "cifar10",
    table_name: str = "exp2_exp3_communication_overhead",
) -> pd.DataFrame:
    """
    Build a plot-ready table of communication overhead metrics.
    """
    comparison = build_exp2_exp3_comparison_table(
        exp2_results_dir=exp2_results_dir,
        exp3_results_dir=exp3_results_dir,
        output_dir=output_dir,
        dataset_name=dataset_name,
        write_artifacts=False,
    )
    table = comparison[
        [
            "experiment",
            "placement",
            "model",
            "communication_overhead_avg",
            "communication_overhead_ratio_pct",
            "communication_overhead_reduction_pct",
        ]
    ].copy()
    _write_table(
        table,
        Path(output_dir),
        table_name,
        caption="Communication overhead in Experiments 2 and 3.",
        label="tab:exp2-exp3-communication-overhead",
    )
    return table


def build_distributed_node_utilization_table(
    results_dirs: Iterable[str | Path] = ALL_DISTRIBUTED_RESULTS_DIRS,
    output_dir: str | Path = "results/thesis_visualizations/distributed",
    dataset_name: str | None = "cifar10",
    table_name: str = "distributed_node_utilization",
    write_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Build a worker-utilization table for all distributed experiments.
    """
    metrics_items = _load_latest_metrics(
        [Path(results_dir) for results_dir in results_dirs],
        dataset_name=dataset_name,
    )
    rows: list[dict[str, Any]] = []
    for _, metrics in metrics_items:
        experiment_id = str(metrics.get("experiment_id", ""))
        experiment = _experiment_label(experiment_id)
        if not _is_distributed_experiment(experiment):
            continue

        for worker_id in WORKER_IDS:
            utilization = _optional_float(metrics.get(f"{worker_id}_node_utilization"))
            if utilization is None:
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "placement": _placement_label(experiment),
                    "model": _model_label(
                        str(metrics.get("model_name", "")),
                        experiment_id,
                    ),
                    "worker": worker_id,
                    "node_utilization": utilization * 100.0,
                }
            )

    if not rows:
        raise FileNotFoundError("No distributed worker-utilization metrics found.")

    table = _sort_table(pd.DataFrame(rows))
    table = _round_table(table, {"node_utilization": 2})
    if write_artifacts:
        _write_table(
            table,
            Path(output_dir),
            table_name,
            caption="Worker utilization across distributed experiments.",
            label="tab:distributed-node-utilization",
        )
    return table


def _load_latest_experiment1_metrics(
    results_dir: Path,
    dataset_name: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    latest_by_experiment: dict[str, tuple[Path, dict[str, Any]]] = {}

    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        metrics = _read_json(metrics_path)
        experiment_id = str(metrics.get("experiment_id", ""))
        if not _is_experiment1_id(experiment_id):
            continue
        if dataset_name and str(metrics.get("dataset_name", "")) != dataset_name:
            continue

        existing = latest_by_experiment.get(experiment_id)
        if existing is None or metrics_path.parent.name > existing[0].parent.name:
            latest_by_experiment[experiment_id] = (metrics_path, metrics)

    return list(latest_by_experiment.values())


def _load_latest_metrics(
    results_dirs: Iterable[Path],
    dataset_name: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
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


def _build_exp2_exp3_rows(
    metrics_items: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    raw_rows = [
        _build_exp2_exp3_row(metrics, metrics_path)
        for metrics_path, metrics in metrics_items
        if _experiment_label(str(metrics.get("experiment_id", ""))) in {"2.0", "3.1", "3.2", "3.3"}
    ]
    baseline_by_model = {
        row["model"]: row for row in raw_rows if row["experiment"] == "2.0"
    }

    for row in raw_rows:
        baseline = baseline_by_model.get(row["model"])
        row = dict(row)
        if baseline is None:
            row["speedup_vs_exp2"] = None
            row["inference_time_reduction_pct"] = None
            row["communication_overhead_reduction_pct"] = None
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

    return rows


def _build_exp2_exp3_row(
    metrics: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any]:
    experiment_id = str(metrics.get("experiment_id", ""))
    experiment = _experiment_label(experiment_id)

    return {
        "experiment": experiment,
        "placement": _placement_label(experiment),
        "model": _model_label(str(metrics.get("model_name", "")), experiment_id),
        "accuracy": _optional_float(metrics.get("accuracy")),
        "inference_time": _resolve_avg_inference_time(metrics, metrics_path),
        "throughput": _optional_float(metrics.get("throughput_samples_per_sec")),
        "communication_overhead_avg": _optional_float(
            metrics.get("communication_overhead_avg_sec")
        ),
        "communication_overhead_ratio_pct": (
            (_optional_float(metrics.get("communication_overhead_ratio_total")) or 0.0)
            * 100.0
        ),
        "worker1_util": _worker_utilization_percent(metrics, "worker1"),
        "worker2_util": _worker_utilization_percent(metrics, "worker2"),
        "worker3_util": _worker_utilization_percent(metrics, "worker3"),
        "route_model_0": metrics.get("route_model_0"),
        "route_model_1": metrics.get("route_model_1"),
    }


def _worker_utilization_percent(
    metrics: dict[str, Any],
    worker_id: str,
) -> float | None:
    value = _optional_float(metrics.get(f"{worker_id}_node_utilization"))
    if value is None:
        return None
    return value * 100.0


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _is_experiment1_id(experiment_id: str) -> bool:
    return any(experiment_id.startswith(prefix) for prefix in EXPERIMENT1_PREFIXES)


def _build_experiment1_row(
    metrics: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any]:
    experiment_id = str(metrics.get("experiment_id", ""))
    model_name = str(metrics.get("model_name", ""))

    return {
        "experiment": _experiment_label(experiment_id),
        "accuracy": _optional_float(metrics.get("accuracy")),
        "inference_time": _resolve_avg_inference_time(metrics, metrics_path),
        "throughput": _optional_float(metrics.get("throughput_samples_per_sec")),
        "model": _model_label(model_name, experiment_id),
        "exit0": _exit_ratio_percent(metrics, 0),
        "exit1": _exit_ratio_percent(metrics, 1),
        "exit2": _exit_ratio_percent(metrics, 2),
        "exit3": _exit_ratio_percent(metrics, 3),
    }


def _experiment_label(experiment_id: str) -> str:
    for prefix, label in EXPERIMENT_LABELS.items():
        if experiment_id.startswith(prefix):
            return label
    return experiment_id


def _placement_label(experiment: str) -> str:
    return PLACEMENT_LABELS.get(experiment, "-")


def _is_distributed_experiment(experiment: str) -> bool:
    return experiment in {"1.3", "2.0", "3.1", "3.2", "3.3"}


def _model_label(model_name: str, experiment_id: str) -> str:
    architecture = _model_architecture(model_name)
    if "exp1_1" in experiment_id or "baseline" in model_name.lower():
        return f"{architecture} Baseline"
    return f"{architecture}EE"


def _model_architecture(model_name: str) -> str:
    normalized = model_name.lower()
    if "resnet34" in normalized:
        return "ResNet34"
    if "resnet18" in normalized:
        return "ResNet18"
    return model_name or "unknown"


def _resolve_avg_inference_time(
    metrics: dict[str, Any],
    metrics_path: Path,
) -> float | None:
    value = metrics.get("avg_inference_time_sec")
    if value is not None:
        return _optional_float(value)

    # Backward compatibility for runs generated before the latency rename.
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


def _exit_ratio_percent(metrics: dict[str, Any], exit_id: int) -> float | None:
    value = metrics.get(f"exit_{exit_id}_ratio")
    if value is None:
        return None
    return float(value) * 100.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _reduction_percent(
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None or baseline == 0.0:
        return None
    return (baseline - candidate) / baseline * 100.0


def _sort_table(table: pd.DataFrame) -> pd.DataFrame:
    sorted_table = table.copy()
    if "experiment" in sorted_table.columns:
        sorted_table["_experiment_sort"] = sorted_table["experiment"].map(
            EXPERIMENT_SORT_ORDER
        )
    else:
        sorted_table["_experiment_sort"] = 0

    if "model" in sorted_table.columns:
        sorted_table["_model_sort"] = sorted_table["model"].map(_model_sort_value)
    else:
        sorted_table["_model_sort"] = 0

    if "worker" in sorted_table.columns:
        sorted_table["_worker_sort"] = sorted_table["worker"].map(
            {worker: i for i, worker in enumerate(WORKER_IDS)}
        )
    else:
        sorted_table["_worker_sort"] = 0

    sorted_table = sorted_table.sort_values(
        by=["_model_sort", "_experiment_sort", "_worker_sort"],
        kind="stable",
    )
    return sorted_table.drop(
        columns=["_experiment_sort", "_model_sort", "_worker_sort"]
    ).reset_index(drop=True)


def _round_table(
    table: pd.DataFrame,
    decimals_by_column: dict[str, int],
) -> pd.DataFrame:
    rounded = table.copy()
    for column, decimals in decimals_by_column.items():
        if column in rounded.columns:
            rounded[column] = rounded[column].round(decimals)
    return rounded


def _write_table(
    table: pd.DataFrame,
    output_dir: Path,
    table_name: str,
    caption: str,
    label: str,
) -> None:
    tables_path = output_dir / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_path / f"{table_name}.csv", index=False, na_rep="-")
    table.to_latex(
        tables_path / f"{table_name}.tex",
        index=False,
        na_rep="-",
        escape=True,
        caption=caption,
        label=label,
    )


def _sort_experiment1_table(table: pd.DataFrame) -> pd.DataFrame:
    sorted_table = table.copy()
    sorted_table["_experiment_sort"] = sorted_table["experiment"].map(
        EXPERIMENT_SORT_ORDER
    )
    sorted_table["_model_sort"] = sorted_table["model"].map(_model_sort_value)
    sorted_table = sorted_table.sort_values(
        by=["_experiment_sort", "_model_sort", "model"],
        kind="stable",
    )
    return sorted_table.drop(columns=["_experiment_sort", "_model_sort"]).reset_index(
        drop=True
    )


def _model_sort_value(model_label: str) -> int:
    normalized = model_label.lower()
    for key, value in MODEL_SORT_ORDER.items():
        if key in normalized:
            return value
    return len(MODEL_SORT_ORDER)


def _round_experiment1_table(table: pd.DataFrame) -> pd.DataFrame:
    rounded = table.copy()
    for column in ["accuracy", "exit0", "exit1", "exit2", "exit3"]:
        rounded[column] = rounded[column].round(2)
    rounded["inference_time"] = rounded["inference_time"].round(4)
    rounded["throughput"] = rounded["throughput"].round(3)
    return rounded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis tables from experiment metrics."
    )
    parser.add_argument(
        "--results-dir",
        default="results/exp1_single_model",
        help="Primary results directory.",
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
        default="results/thesis_visualizations/exp1_single_model",
        help="Directory where generated table artifacts will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="cifar10",
        help="Dataset filter. Use an empty string to include all datasets.",
    )
    parser.add_argument(
        "--table-name",
        default="experiment1_core_metrics",
        help="Base filename for the generated CSV and LaTeX table.",
    )
    parser.add_argument(
        "--table",
        choices=(
            "experiment1_core",
            "exp2_exp3_comparison",
            "exp2_exp3_speedup",
            "exp2_best_exp3_worker_utilization",
            "exp2_exp3_communication_overhead",
            "distributed_node_utilization",
            "all_exp2_exp3",
        ),
        default="experiment1_core",
        help="Table to generate.",
    )
    parser.add_argument(
        "--best-experiment",
        default="3.2",
        help="Best Experiment 3 label used for Exp2-vs-best utilization tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name if args.dataset_name else None
    generated_tables: list[tuple[str, pd.DataFrame]] = []

    if args.table == "experiment1_core":
        generated_tables.append(
            (
                "experiment1_core",
                build_experiment1_core_table(
                    results_dir=args.results_dir,
                    output_dir=args.output_dir,
                    dataset_name=dataset_name,
                    table_name=args.table_name,
                ),
            )
        )
    elif args.table in {"exp2_exp3_comparison", "all_exp2_exp3"}:
        generated_tables.append(
            (
                "exp2_exp3_comparison",
                build_exp2_exp3_comparison_table(
                    exp2_results_dir=args.exp2_results_dir,
                    exp3_results_dir=args.exp3_results_dir,
                    output_dir=args.output_dir,
                    dataset_name=dataset_name,
                ),
            )
        )

    if args.table in {"exp2_exp3_speedup", "all_exp2_exp3"}:
        generated_tables.append(
            (
                "exp2_exp3_speedup",
                build_exp2_exp3_speedup_table(
                    exp2_results_dir=args.exp2_results_dir,
                    exp3_results_dir=args.exp3_results_dir,
                    output_dir=args.output_dir,
                    dataset_name=dataset_name,
                ),
            )
        )

    if args.table in {"exp2_best_exp3_worker_utilization", "all_exp2_exp3"}:
        generated_tables.append(
            (
                "exp2_best_exp3_worker_utilization",
                build_exp2_best_exp3_worker_utilization_table(
                    exp2_results_dir=args.exp2_results_dir,
                    exp3_results_dir=args.exp3_results_dir,
                    output_dir=args.output_dir,
                    dataset_name=dataset_name,
                    best_experiment=args.best_experiment,
                ),
            )
        )

    if args.table in {"exp2_exp3_communication_overhead", "all_exp2_exp3"}:
        generated_tables.append(
            (
                "exp2_exp3_communication_overhead",
                build_exp2_exp3_communication_overhead_table(
                    exp2_results_dir=args.exp2_results_dir,
                    exp3_results_dir=args.exp3_results_dir,
                    output_dir=args.output_dir,
                    dataset_name=dataset_name,
                ),
            )
        )

    if args.table in {"distributed_node_utilization", "all_exp2_exp3"}:
        generated_tables.append(
            (
                "distributed_node_utilization",
                build_distributed_node_utilization_table(
                    results_dirs=ALL_DISTRIBUTED_RESULTS_DIRS,
                    output_dir=args.output_dir,
                    dataset_name=dataset_name,
                ),
            )
        )

    for name, table in generated_tables:
        print(f"{name}:")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
