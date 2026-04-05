from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.visualization.summary import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    extract_worker_ids,
    load_summary_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis comparison tables from Experiment 1 metrics."
    )
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    with open(out_dir / f"{stem}.md", "w", encoding="utf-8") as handle:
        handle.write(_dataframe_to_markdown(df))
        handle.write("\n")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows: list[str] = [header, separator]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def _core_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "topology_label",
        "accuracy",
        "throughput_samples_per_sec",
        "avg_latency_sec",
        "p95_latency_sec",
        "p99_latency_sec",
        "total_inference_time_sec",
        "num_samples",
    ]
    table = summary_df[columns].copy()
    return table.rename(
        columns={
            "experiment_id": "experiment",
            "topology_label": "topology",
            "accuracy": "accuracy_pct",
            "throughput_samples_per_sec": "throughput_samples_per_sec",
            "avg_latency_sec": "avg_latency_sec",
            "p95_latency_sec": "p95_latency_sec",
            "p99_latency_sec": "p99_latency_sec",
            "total_inference_time_sec": "total_time_sec",
            "num_samples": "samples",
        }
    )


def _energy_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "experiment": row["experiment_id"],
                "topology": row["topology_label"],
                "master_energy_kWh": row.get("master_energy_kWh", row.get("energy_kWh")),
                "workers_energy_kWh_total": row.get("workers_energy_kWh_total"),
                "system_energy_kWh_total": row.get("system_energy_kWh_total", row.get("energy_kWh")),
                "master_carbon_kg": row.get("master_carbon_kg", row.get("carbon_kg")),
                "workers_carbon_kg_total": row.get("workers_carbon_kg_total"),
                "system_carbon_kg_total": row.get("system_carbon_kg_total", row.get("carbon_kg")),
            }
        )
    return pd.DataFrame(rows)


def _exit_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "topology_label",
        "exit_0_ratio",
        "exit_1_ratio",
        "exit_2_ratio",
        "exit_3_ratio",
    ]
    return summary_df[columns].rename(
        columns={
            "experiment_id": "experiment",
            "topology_label": "topology",
        }
    )


def _worker_breakdown_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        worker_ids = extract_worker_ids(row.to_dict())
        for worker_id in worker_ids:
            rows.append(
                {
                    "experiment": row["experiment_id"],
                    "topology": row["topology_label"],
                    "worker_id": worker_id,
                    "compute_time_total_sec": row.get(f"{worker_id}_compute_time_total_sec"),
                    "compute_time_avg_sec": row.get(f"{worker_id}_compute_time_avg_sec"),
                    "request_bytes_total": row.get(f"{worker_id}_request_bytes_total"),
                    "response_bytes_total": row.get(f"{worker_id}_response_bytes_total"),
                    "carbon_kg": row.get(f"{worker_id}_carbon_kg"),
                    "energy_kWh": row.get(f"{worker_id}_energy_kWh"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    summary_df = load_summary_dataframe(Path(args.results_dir))
    output_dir = Path(args.output_dir) / "tables"

    _write_table(_core_table(summary_df), output_dir, "core_metrics")
    _write_table(_energy_table(summary_df), output_dir, "energy_metrics")
    _write_table(_exit_table(summary_df), output_dir, "exit_distribution")

    worker_table = _worker_breakdown_table(summary_df)
    if not worker_table.empty:
        _write_table(worker_table, output_dir, "worker_breakdown")


if __name__ == "__main__":
    main()
