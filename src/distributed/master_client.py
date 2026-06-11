from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from codecarbon import EmissionsTracker
from tqdm import tqdm

from src.data.loaders import data_loader
from src.distributed.client.fastapi_client import start_monitoring, stop_monitoring
from src.inference.partition_runner import run_chained_inference
from src.metrics.accuracy import compute_accuracy
from src.metrics.exits import (
    initialize_exit_counts,
    summarize_exit_counts,
    update_exit_counts,
)
from src.metrics.inference_time import (
    compute_duration_stats,
    compute_inference_time_stats,
    compute_throughput,
    compute_total_inference_time,
)
from src.metrics.utilization import compute_node_utilization
from src.utils.config import load_experiment_bundle, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master coordinator for multi-worker distributed EE inference."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def _find_worker_cfg(system_cfg: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker_cfg in system_cfg.get("workers", []):
        if worker_cfg.get("worker_id") == worker_id:
            return worker_cfg
    raise ValueError(f"Worker '{worker_id}' not found in system config")


def _get_ordered_worker_cfgs(system_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    pipeline_order = system_cfg.get("pipeline_order")
    if pipeline_order:
        return [
            _find_worker_cfg(system_cfg, str(worker_id)) for worker_id in pipeline_order
        ]

    workers = list(system_cfg.get("workers", []))
    workers.sort(key=lambda w: int(w.get("partition_id", 0)))
    return workers


def _make_stage_metric_maps(
    worker_cfgs: list[dict[str, Any]], default_value: float | int
) -> dict[str, float | int]:
    return {str(worker_cfg["worker_id"]): default_value for worker_cfg in worker_cfgs}


def _start_worker_monitors(
    worker_cfgs: list[dict[str, Any]],
    timeout_sec: float,
) -> None:
    for worker_cfg in worker_cfgs:
        start_monitoring(worker_cfg=worker_cfg, timeout_sec=timeout_sec)


def _stop_worker_monitors(
    worker_cfgs: list[dict[str, Any]],
    timeout_sec: float,
) -> dict[str, dict[str, float | None]]:
    monitoring_results: dict[str, dict[str, float | None]] = {}
    for worker_cfg in worker_cfgs:
        worker_id = str(worker_cfg["worker_id"])
        response = stop_monitoring(worker_cfg=worker_cfg, timeout_sec=timeout_sec)
        monitoring_results[worker_id] = {
            "carbon_kg": (
                float(response["carbon_kg"])
                if response.get("carbon_kg") is not None
                else None
            ),
            "energy_kWh": (
                float(response["energy_kWh"])
                if response.get("energy_kWh") is not None
                else None
            ),
        }
    return monitoring_results


def save_results(
    output_dir: str,
    summary: dict[str, Any],
    per_sample_df: pd.DataFrame,
    config_bundle: dict[str, Any],
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    per_sample_df.to_csv(out_path / "inference_times.csv", index=False)

    with open(out_path / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(config_bundle, f, indent=2)


def evaluate_distributed_ee(
    dataset_cfg: dict[str, Any],
    system_cfg: dict[str, Any],
    data_dir: str,
    batch_size: int,
    warmup_samples: int,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if batch_size != 1:
        raise ValueError(
            "Distributed EE experiment currently supports batch_size=1 only."
        )

    worker_cfgs = _get_ordered_worker_cfgs(system_cfg)
    num_stages = len(worker_cfgs)

    if num_stages != 3:
        raise ValueError(
            f"Unsupported number of workers/stages: {num_stages}. Expected 3."
        )

    entry_worker_cfg = worker_cfgs[0]
    num_workers = dataset_cfg.get("loader", {}).get("num_workers", 0)

    test_loader = data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        test=True,
        num_workers=num_workers,
        dataset_config=dataset_cfg,
    )

    dataset_size = (
        len(test_loader.dataset) if hasattr(test_loader, "dataset") else None  # type: ignore
    )
    target_total = (
        min(max_samples, dataset_size)
        if (max_samples is not None and dataset_size is not None)
        else max_samples
    )
    if target_total is None:
        target_total = dataset_size

    timeout_sec = float(system_cfg.get("runtime", {}).get("request_timeout_sec", 30.0))

    if warmup_samples > 0:
        warmup_count = 0
        with torch.no_grad():
            for images, _ in test_loader:
                _ = run_chained_inference(
                    image_tensor=images.cpu(),
                    sample_id=warmup_count,
                    entry_worker_cfg=entry_worker_cfg,
                    timeout_sec=timeout_sec,
                )
                warmup_count += images.size(0)
                if warmup_count >= warmup_samples:
                    break

        test_loader = data_loader(
            data_dir=data_dir,
            batch_size=batch_size,
            test=True,
            num_workers=num_workers,
            dataset_config=dataset_cfg,
        )

    correct = 0
    total = 0
    inference_times: list[float] = []
    per_sample_rows: list[dict[str, Any]] = []
    exit_counts = initialize_exit_counts(4)

    remote_compute_total = 0.0
    communication_overheads: list[float] = []
    communication_overhead_ratios: list[float] = []

    worker_compute_totals = _make_stage_metric_maps(worker_cfgs, 0.0)

    tracker = EmissionsTracker(
        measure_power_secs=1,
        log_level="critical",
    )
    worker_monitoring_results: dict[str, dict[str, float | None]] = {}

    _start_worker_monitors(worker_cfgs=worker_cfgs, timeout_sec=timeout_sec)
    tracker.start()
    experiment_start = time.time()
    inferred_samples = 0
    progress = tqdm(
        total=target_total,
        desc="Distributed inference",
        unit="sample",
        disable=not show_progress,
        dynamic_ncols=True,
    )

    try:
        with torch.no_grad():
            sample_index = 0
            for images, labels in test_loader:
                if max_samples is not None and inferred_samples >= max_samples:
                    break

                labels = labels.cpu()

                start = time.time()
                distributed_output = run_chained_inference(
                    image_tensor=images.cpu(),
                    sample_id=sample_index,
                    entry_worker_cfg=entry_worker_cfg,
                    timeout_sec=timeout_sec,
                )
                end = time.time()

                predicted_class = int(distributed_output["predicted_class"])
                exit_id = int(distributed_output["exit_id"])

                inference_time = end - start
                inference_times.append(inference_time)

                remote_compute_time_sec = float(
                    distributed_output["remote_compute_time_sec"]
                )
                communication_overhead_sec = inference_time - remote_compute_time_sec
                communication_overhead_ratio = (
                    communication_overhead_sec / inference_time
                    if inference_time > 0.0
                    else 0.0
                )

                remote_compute_total += remote_compute_time_sec
                communication_overheads.append(communication_overhead_sec)
                communication_overhead_ratios.append(communication_overhead_ratio)

                worker_compute_times = distributed_output["worker_compute_times"]

                for worker_cfg in worker_cfgs:
                    worker_id = str(worker_cfg["worker_id"])
                    worker_compute_totals[worker_id] = float(
                        worker_compute_totals[worker_id]
                    ) + float(worker_compute_times.get(worker_id, 0.0))

                update_exit_counts(exit_counts, exit_id)

                label_value = int(labels[0].item())
                is_correct = int(predicted_class == label_value)
                correct += is_correct
                total += 1

                row: dict[str, Any] = {
                    "sample_index": sample_index,
                    "batch_size": int(labels.size(0)),
                    "inference_time_sec": float(inference_time),
                    "predicted_class": predicted_class,
                    "true_class": label_value,
                    "correct": is_correct,
                    "exit_id": exit_id,
                    "confidence": distributed_output.get("confidence"),
                    "remote_compute_time_sec": remote_compute_time_sec,
                    "communication_overhead_sec": communication_overhead_sec,
                    "communication_overhead_ratio": communication_overhead_ratio,
                    "path": "->".join(distributed_output.get("path", [])),
                }

                for worker_cfg in worker_cfgs:
                    worker_id = str(worker_cfg["worker_id"])
                    row[f"{worker_id}_compute_time_sec"] = float(
                        worker_compute_times.get(worker_id, 0.0)
                    )

                per_sample_rows.append(row)
                sample_index += 1
                inferred_samples += 1
                progress.update(1)
    finally:
        progress.close()
        experiment_end = time.time()
        tracker.stop()
        worker_monitoring_results = _stop_worker_monitors(
            worker_cfgs=worker_cfgs,
            timeout_sec=timeout_sec,
        )

    total_inference_time_sec = compute_total_inference_time(
        experiment_start, experiment_end
    )
    inference_time_stats = compute_inference_time_stats(inference_times)
    communication_overhead_stats = compute_duration_stats(communication_overheads)
    throughput = compute_throughput(total, total_inference_time_sec)
    node_utilization = compute_node_utilization(
        inference_time_stats["busy_time_sec"],
        total_inference_time_sec,
    )
    accuracy = compute_accuracy(correct, total)

    emissions_data = tracker._prepare_emissions_data()
    carbon_kg = emissions_data.emissions
    energy_kwh = emissions_data.energy_consumed

    results: dict[str, Any] = {
        "mode": f"distributed_early_exit_{num_stages}workers",
        "accuracy": accuracy,
        "num_correct": int(correct),
        "num_samples": int(total),
        "total_inference_time_sec": float(total_inference_time_sec),
        "throughput_samples_per_sec": float(throughput),
        "master_node_utilization": float(node_utilization),
        "master_carbon_kg": float(carbon_kg) if carbon_kg is not None else None,
        "master_energy_kWh": float(energy_kwh) if energy_kwh is not None else None,
        "remote_compute_time_total_sec": float(remote_compute_total),
        "remote_compute_time_avg_sec": (
            float(remote_compute_total / total) if total > 0 else 0.0
        ),
        "communication_overhead_total_sec": float(
            communication_overhead_stats["duration_sum_sec"]
        ),
        "communication_overhead_avg_sec": float(
            communication_overhead_stats["avg_duration_sec"]
        ),
        "communication_overhead_std_sec": float(
            communication_overhead_stats["std_duration_sec"]
        ),
        "communication_overhead_min_sec": float(
            communication_overhead_stats["min_duration_sec"]
        ),
        "communication_overhead_max_sec": float(
            communication_overhead_stats["max_duration_sec"]
        ),
        "communication_overhead_ratio_avg": (
            float(sum(communication_overhead_ratios) / total) if total > 0 else 0.0
        ),
        "communication_overhead_ratio_total": (
            float(
                communication_overhead_stats["duration_sum_sec"]
                / sum(inference_times)
            )
            if sum(inference_times) > 0.0
            else 0.0
        ),
    }
    results.update(inference_time_stats)
    results.update(summarize_exit_counts(exit_counts, total))

    worker_carbon_total = 0.0
    worker_energy_total = 0.0

    for worker_cfg in worker_cfgs:
        worker_id = str(worker_cfg["worker_id"])
        compute_total = float(worker_compute_totals[worker_id])
        worker_node_utilization = compute_node_utilization(
            compute_total,
            total_inference_time_sec,
        )
        worker_monitoring = worker_monitoring_results.get(worker_id, {})
        worker_carbon_kg = worker_monitoring.get("carbon_kg")
        worker_energy_kwh = worker_monitoring.get("energy_kWh")

        results[f"{worker_id}_compute_time_total_sec"] = compute_total
        results[f"{worker_id}_compute_time_avg_sec"] = (
            float(compute_total / total) if total > 0 else 0.0
        )
        results[f"{worker_id}_node_utilization"] = float(worker_node_utilization)
        results[f"{worker_id}_carbon_kg"] = worker_carbon_kg
        results[f"{worker_id}_energy_kWh"] = worker_energy_kwh

        if worker_carbon_kg is not None:
            worker_carbon_total += float(worker_carbon_kg)
        if worker_energy_kwh is not None:
            worker_energy_total += float(worker_energy_kwh)

    results["workers_carbon_kg_total"] = float(worker_carbon_total)
    results["workers_energy_kWh_total"] = float(worker_energy_total)
    results["system_carbon_kg_total"] = float(
        worker_carbon_total + (float(carbon_kg) if carbon_kg is not None else 0.0)
    )
    results["system_energy_kWh_total"] = float(
        worker_energy_total + (float(energy_kwh) if energy_kwh is not None else 0.0)
    )

    per_sample_df = pd.DataFrame(per_sample_rows)
    return results, per_sample_df


def main() -> None:
    args = parse_args()

    bundle = load_experiment_bundle(args.config)
    experiment_cfg = bundle["experiment_config"]
    dataset_cfg = bundle["dataset_config"]
    model_cfg = bundle["model_config"]
    system_cfg = bundle["system_config"]
    repo_root = bundle["repo_root"]

    output_dir = resolve_path(experiment_cfg["output"]["dir"], repo_root)
    data_dir = resolve_path(dataset_cfg["root"], repo_root)

    batch_size = int(experiment_cfg.get("runtime", {}).get("batch_size", 1))
    warmup_samples = int(experiment_cfg.get("runtime", {}).get("warmup_samples", 0))

    summary, per_sample_df = evaluate_distributed_ee(
        dataset_cfg=dataset_cfg,
        system_cfg=system_cfg,
        data_dir=str(data_dir),
        batch_size=batch_size,
        warmup_samples=warmup_samples,
    )

    weights_path = None
    if isinstance(model_cfg.get("weights"), dict):
        weights_path = resolve_path(model_cfg["weights"].get("path"), repo_root)

    summary["experiment_id"] = experiment_cfg.get("experiment", {}).get("id")
    summary["experiment_name"] = experiment_cfg.get("experiment", {}).get("name")
    summary["dataset_name"] = dataset_cfg.get("name")
    summary["model_name"] = model_cfg.get("name")
    summary["system_name"] = system_cfg.get("system_name")
    summary["weights_path"] = weights_path
    summary["data_dir"] = str(data_dir)
    summary["output_dir"] = str(output_dir)

    save_results(str(output_dir), summary, per_sample_df, bundle)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
