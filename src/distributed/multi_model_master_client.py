from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from codecarbon import EmissionsTracker

from src.data.loaders import data_loader
from src.distributed.client.fastapi_client import start_monitoring, stop_monitoring
from src.distributed.master_client import (
    _get_ordered_worker_cfgs,
    _make_stage_metric_maps,
)
from src.distributed.runtime.worker_runtime import (
    resolve_model_instance_ids,
    resolve_placement_routes,
    resolve_worker_assignments,
    validate_placement_config,
)
from src.inference.partition_runner import run_chained_inference
from src.metrics.accuracy import compute_accuracy
from src.metrics.exits import (
    initialize_exit_counts,
    summarize_exit_counts,
    update_exit_counts,
)
from src.metrics.latency import (
    compute_latency_stats,
    compute_throughput,
    compute_total_inference_time,
)
from src.metrics.network import compute_network_delta, read_network_bytes
from src.metrics.utilization import compute_node_utilization
from src.utils.config import load_experiment_bundle, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master coordinator for homogeneous multi-model distributed EE inference."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


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


def _build_test_loader(
    *,
    dataset_cfg: dict[str, Any],
    data_dir: str,
    batch_size: int,
):
    loader_cfg = dataset_cfg.get("loader", {})
    return data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        test=True,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        dataset_config=dataset_cfg,
    )


def _get_loader_dataset_size(loader: Any) -> int | None:
    return len(loader.dataset) if hasattr(loader, "dataset") else None


def _run_one_inference(
    *,
    model_instance_id: str,
    sample_index: int,
    image_tensor: torch.Tensor,
    label_value: int,
    entry_worker_cfg: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    start = time.time()
    distributed_output = run_chained_inference(
        image_tensor=image_tensor,
        sample_id=sample_index,
        entry_worker_cfg=entry_worker_cfg,
        timeout_sec=timeout_sec,
        model_instance_id=model_instance_id,
    )
    latency = time.time() - start

    predicted_class = int(distributed_output["predicted_class"])
    remote_compute_time_sec = float(distributed_output["remote_compute_time_sec"])
    communication_overhead_sec = latency - remote_compute_time_sec
    communication_overhead_ratio = (
        communication_overhead_sec / latency if latency > 0.0 else 0.0
    )

    path = list(distributed_output.get("path", []))

    return {
        "model_instance_id": model_instance_id,
        "sample_index": sample_index,
        "batch_size": 1,
        "latency_sec": float(latency),
        "predicted_class": predicted_class,
        "true_class": label_value,
        "correct": int(predicted_class == label_value),
        "exit_id": int(distributed_output["exit_id"]),
        "confidence": distributed_output.get("confidence"),
        "protocol_bytes": int(distributed_output["protocol_bytes"]),
        "remote_compute_time_sec": remote_compute_time_sec,
        "communication_overhead_sec": communication_overhead_sec,
        "communication_overhead_ratio": communication_overhead_ratio,
        "path": "->".join(path),
        "entry_worker_id": str(entry_worker_cfg["worker_id"]),
        "terminal_worker_id": str(path[-1]) if path else "",
        "assigned_partition_id": int(entry_worker_cfg.get("partition_id", 0)),
        "worker_compute_times": distributed_output["worker_compute_times"],
        "stage_request_bytes": distributed_output["stage_request_bytes"],
        "stage_response_bytes": distributed_output["stage_response_bytes"],
    }


def _run_warmup(
    *,
    model_instance_ids: list[str],
    sample_records: list[tuple[int, torch.Tensor, int]],
    warmup_samples: int,
    entry_worker_cfgs_by_model: dict[str, dict[str, Any]],
    timeout_sec: float,
    concurrency: int,
) -> None:
    if warmup_samples <= 0:
        return

    warmup_records = sample_records[:warmup_samples]
    if not warmup_records:
        return

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for model_instance_id in model_instance_ids:
            for sample_index, image_tensor, label_value in warmup_records:
                futures.append(
                    executor.submit(
                        _run_one_inference,
                        model_instance_id=model_instance_id,
                        sample_index=sample_index,
                        image_tensor=image_tensor,
                        label_value=label_value,
                        entry_worker_cfg=entry_worker_cfgs_by_model[
                            model_instance_id
                        ],
                        timeout_sec=timeout_sec,
                    )
                )

        for future in as_completed(futures):
            future.result()


def _build_worker_cfg_by_id(
    worker_cfgs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(worker_cfg["worker_id"]): worker_cfg for worker_cfg in worker_cfgs}


def _resolve_entry_worker_cfgs_by_model(
    *,
    experiment_cfg: dict[str, Any],
    worker_cfgs: list[dict[str, Any]],
    model_instance_ids: list[str],
) -> dict[str, dict[str, Any]]:
    routes = resolve_placement_routes(experiment_cfg)
    if not routes:
        return {model_instance_id: worker_cfgs[0] for model_instance_id in model_instance_ids}

    worker_cfg_by_id = _build_worker_cfg_by_id(worker_cfgs)
    entry_cfgs: dict[str, dict[str, Any]] = {}
    for model_instance_id in model_instance_ids:
        route = routes.get(model_instance_id)
        if not route:
            raise ValueError(f"No placement route configured for {model_instance_id}")
        first_entry = route[0]
        if first_entry.worker_id not in worker_cfg_by_id:
            raise ValueError(
                f"Route for {model_instance_id} references unknown entry worker "
                f"{first_entry.worker_id}"
            )
        cfg = dict(worker_cfg_by_id[first_entry.worker_id])
        cfg["partition_id"] = int(first_entry.partition_id)
        cfg["next_worker_id"] = str(route[1].worker_id) if len(route) > 1 else None
        entry_cfgs[model_instance_id] = cfg
    return entry_cfgs


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

    per_sample_df.to_csv(out_path / "latencies.csv", index=False)

    with open(out_path / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(config_bundle, f, indent=2)


def evaluate_multi_model_distributed_ee(
    *,
    experiment_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    system_cfg: dict[str, Any],
    data_dir: str,
    repo_root: str,
    batch_size: int,
    max_samples_per_model: int | None,
    show_progress: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    worker_cfgs = _get_ordered_worker_cfgs(system_cfg)
    validate_placement_config(
        experiment_cfg=experiment_cfg,
        system_cfg=system_cfg,
    )
    num_workers = len(worker_cfgs)
    if num_workers not in {2, 3}:
        raise ValueError(
            f"Unsupported number of workers/stages: {num_workers}. Expected 2 or 3."
        )

    model_instance_ids = resolve_model_instance_ids(
        experiment_cfg=experiment_cfg,
        system_cfg=system_cfg,
    )
    placement_enabled = bool(resolve_placement_routes(experiment_cfg))
    expected_instances = max(num_workers - 1, 1)
    if not placement_enabled and len(model_instance_ids) != expected_instances:
        raise ValueError(
            f"Experiment 2 expects N-1 model instances for N workers. "
            f"Got {len(model_instance_ids)} model instances for {num_workers} workers."
        )

    runtime_cfg = experiment_cfg.get("runtime", {})
    timeout_sec = float(system_cfg.get("runtime", {}).get("request_timeout_sec", 30.0))
    concurrency = int(runtime_cfg.get("concurrency", len(model_instance_ids)))
    concurrency = max(concurrency, 1)
    warmup_samples = int(runtime_cfg.get("warmup_samples", 0))

    warmup_loader = _build_test_loader(
        dataset_cfg=dataset_cfg,
        data_dir=data_dir,
        batch_size=batch_size,
    )
    dataset_size = _get_loader_dataset_size(warmup_loader)
    target_samples_per_model = (
        min(max_samples_per_model, dataset_size)
        if (max_samples_per_model is not None and dataset_size is not None)
        else max_samples_per_model
    )
    if target_samples_per_model is None:
        target_samples_per_model = dataset_size
    if target_samples_per_model == 0:
        raise ValueError("No samples available for Experiment 2 evaluation.")

    warmup_records: list[tuple[int, torch.Tensor, int]] = []
    if warmup_samples > 0:
        for sample_index, (images, labels) in enumerate(warmup_loader):
            warmup_records.append(
                (sample_index, images.cpu(), int(labels.cpu()[0].item()))
            )
            if len(warmup_records) >= warmup_samples:
                break

    entry_worker_cfgs_by_model = _resolve_entry_worker_cfgs_by_model(
        experiment_cfg=experiment_cfg,
        worker_cfgs=worker_cfgs,
        model_instance_ids=model_instance_ids,
    )
    _run_warmup(
        model_instance_ids=model_instance_ids,
        sample_records=warmup_records,
        warmup_samples=warmup_samples,
        entry_worker_cfgs_by_model=entry_worker_cfgs_by_model,
        timeout_sec=timeout_sec,
        concurrency=concurrency,
    )

    test_loader = _build_test_loader(
        dataset_cfg=dataset_cfg,
        data_dir=data_dir,
        batch_size=batch_size,
    )

    master_monitor_cfg = system_cfg.get("monitoring", {})
    network_interface = master_monitor_cfg.get("network_interface", None)

    net_before = read_network_bytes(interface=network_interface)
    tracker = EmissionsTracker(measure_power_secs=1, log_level="critical")
    worker_monitoring_results: dict[str, dict[str, float | None]] = {}

    _start_worker_monitors(worker_cfgs=worker_cfgs, timeout_sec=timeout_sec)
    tracker.start()
    experiment_start = time.time()

    rows: list[dict[str, Any]] = []
    total_jobs = (
        len(model_instance_ids) * target_samples_per_model
        if target_samples_per_model is not None
        else None
    )

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending: set[Future[dict[str, Any]]] = set()
            completed = 0

            def collect_done(done_futures: set[Future[dict[str, Any]]]) -> None:
                nonlocal completed
                for done_future in done_futures:
                    row = done_future.result()
                    rows.append(row)
                    completed += 1
                    if show_progress:
                        if total_jobs is None:
                            print(
                                f"\rInferred {completed} model-samples",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\rInferred {completed}/{total_jobs} model-samples",
                                end="",
                                flush=True,
                            )

            max_pending = max(concurrency * 2, 1)
            sample_count = 0
            for sample_index, (images, labels) in enumerate(test_loader):
                if (
                    max_samples_per_model is not None
                    and sample_count >= max_samples_per_model
                ):
                    break

                image_tensor = images.cpu()
                label_value = int(labels.cpu()[0].item())

                for model_instance_id in model_instance_ids:
                    pending.add(
                        executor.submit(
                            _run_one_inference,
                            model_instance_id=model_instance_id,
                            sample_index=sample_index,
                            image_tensor=image_tensor,
                            label_value=label_value,
                            entry_worker_cfg=entry_worker_cfgs_by_model[
                                model_instance_id
                            ],
                            timeout_sec=timeout_sec,
                        )
                    )
                    if len(pending) >= max_pending:
                        done, pending = wait(
                            pending,
                            return_when=FIRST_COMPLETED,
                        )
                        collect_done(done)

                sample_count += 1

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect_done(done)

            if show_progress:
                print()
    finally:
        experiment_end = time.time()
        tracker.stop()
        worker_monitoring_results = _stop_worker_monitors(
            worker_cfgs=worker_cfgs,
            timeout_sec=timeout_sec,
        )

    net_after = read_network_bytes(interface=network_interface)
    rows.sort(key=lambda row: (row["model_instance_id"], row["sample_index"]))

    total = len(rows)
    latencies = [float(row["latency_sec"]) for row in rows]
    communication_overheads = [
        float(row["communication_overhead_sec"]) for row in rows
    ]
    communication_overhead_ratios = [
        float(row["communication_overhead_ratio"]) for row in rows
    ]
    correct = sum(int(row["correct"]) for row in rows)
    protocol_bytes_total = sum(int(row["protocol_bytes"]) for row in rows)
    remote_compute_total = sum(float(row["remote_compute_time_sec"]) for row in rows)

    worker_compute_totals = _make_stage_metric_maps(worker_cfgs, 0.0)
    stage_request_totals = _make_stage_metric_maps(worker_cfgs, 0)
    stage_response_totals = _make_stage_metric_maps(worker_cfgs, 0)
    per_model_worker_compute: dict[str, dict[str, float]] = {
        model_instance_id: {
            str(worker_cfg["worker_id"]): 0.0 for worker_cfg in worker_cfgs
        }
        for model_instance_id in model_instance_ids
    }

    exit_counts = initialize_exit_counts(4)
    per_model_rows: dict[str, list[dict[str, Any]]] = {
        model_instance_id: [] for model_instance_id in model_instance_ids
    }
    per_model_exit_counts = {
        model_instance_id: initialize_exit_counts(4)
        for model_instance_id in model_instance_ids
    }

    flattened_rows: list[dict[str, Any]] = []
    for row in rows:
        model_instance_id = str(row["model_instance_id"])
        worker_compute_times = row.pop("worker_compute_times")
        stage_request_bytes = row.pop("stage_request_bytes")
        stage_response_bytes = row.pop("stage_response_bytes")

        update_exit_counts(exit_counts, int(row["exit_id"]))
        update_exit_counts(per_model_exit_counts[model_instance_id], int(row["exit_id"]))
        per_model_rows[model_instance_id].append(row)

        for worker_cfg in worker_cfgs:
            worker_id = str(worker_cfg["worker_id"])
            compute_time = float(worker_compute_times.get(worker_id, 0.0))
            request_bytes = int(stage_request_bytes.get(worker_id, 0))
            response_bytes = int(stage_response_bytes.get(worker_id, 0))

            worker_compute_totals[worker_id] = float(
                worker_compute_totals[worker_id]
            ) + compute_time
            stage_request_totals[worker_id] = int(
                stage_request_totals[worker_id]
            ) + request_bytes
            stage_response_totals[worker_id] = int(
                stage_response_totals[worker_id]
            ) + response_bytes
            per_model_worker_compute[model_instance_id][worker_id] += compute_time

            row[f"{worker_id}_compute_time_sec"] = compute_time
            row[f"{worker_id}_request_bytes"] = request_bytes
            row[f"{worker_id}_response_bytes"] = response_bytes

        flattened_rows.append(row)

    total_inference_time_sec = compute_total_inference_time(
        experiment_start, experiment_end
    )
    latency_stats = compute_latency_stats(latencies)
    communication_overhead_stats = compute_latency_stats(communication_overheads)
    throughput = compute_throughput(total, total_inference_time_sec)
    node_utilization = compute_node_utilization(
        latency_stats["busy_time_sec"],
        total_inference_time_sec,
    )
    network_stats = compute_network_delta(net_before, net_after)

    emissions_data = tracker._prepare_emissions_data()
    carbon_kg = emissions_data.emissions
    energy_kwh = emissions_data.energy_consumed

    results: dict[str, Any] = {
        "mode": f"distributed_multi_model_ee_{num_workers}workers",
        "num_workers": int(num_workers),
        "num_model_instances": int(len(model_instance_ids)),
        "model_instance_ids": ",".join(model_instance_ids),
        "accuracy": compute_accuracy(correct, total),
        "num_correct": int(correct),
        "num_samples": int(total),
        "samples_per_model": (
            int(total / len(model_instance_ids)) if model_instance_ids else 0
        ),
        "total_inference_time_sec": float(total_inference_time_sec),
        "throughput_samples_per_sec": float(throughput),
        "master_node_utilization": float(node_utilization),
        "master_carbon_kg": float(carbon_kg) if carbon_kg is not None else None,
        "master_energy_kWh": float(energy_kwh) if energy_kwh is not None else None,
        "master_network_rx_bytes": int(network_stats["rx_bytes"]),
        "master_network_tx_bytes": int(network_stats["tx_bytes"]),
        "master_network_total_bytes": int(network_stats["total_bytes"]),
        "protocol_bytes_total": int(protocol_bytes_total),
        "avg_protocol_bytes_per_sample": (
            float(protocol_bytes_total / total) if total > 0 else 0.0
        ),
        "remote_compute_time_total_sec": float(remote_compute_total),
        "remote_compute_time_avg_sec": (
            float(remote_compute_total / total) if total > 0 else 0.0
        ),
        "communication_overhead_total_sec": float(
            communication_overhead_stats["busy_time_sec"]
        ),
        "communication_overhead_avg_sec": float(
            communication_overhead_stats["avg_latency_sec"]
        ),
        "communication_overhead_std_sec": float(
            communication_overhead_stats["std_latency_sec"]
        ),
        "communication_overhead_min_sec": float(
            communication_overhead_stats["min_latency_sec"]
        ),
        "communication_overhead_max_sec": float(
            communication_overhead_stats["max_latency_sec"]
        ),
        "communication_overhead_p50_sec": float(
            communication_overhead_stats["p50_latency_sec"]
        ),
        "communication_overhead_p95_sec": float(
            communication_overhead_stats["p95_latency_sec"]
        ),
        "communication_overhead_p99_sec": float(
            communication_overhead_stats["p99_latency_sec"]
        ),
        "communication_overhead_ratio_avg": (
            float(sum(communication_overhead_ratios) / total) if total > 0 else 0.0
        ),
        "communication_overhead_ratio_total": (
            float(communication_overhead_stats["busy_time_sec"] / sum(latencies))
            if sum(latencies) > 0.0
            else 0.0
        ),
    }
    results.update(latency_stats)
    results.update(summarize_exit_counts(exit_counts, total))

    placement_routes = resolve_placement_routes(experiment_cfg)
    placement_assignments = resolve_worker_assignments(experiment_cfg)
    if placement_routes:
        runtime_cfg = experiment_cfg.get("runtime", {})
        placement_cfg = experiment_cfg.get("placement", {})
        results["placement_strategy"] = runtime_cfg.get("placement_strategy")
        results["spare_worker_id"] = placement_cfg.get("spare_worker_id")
        for model_instance_id, route in placement_routes.items():
            results[f"route_{model_instance_id}"] = "->".join(
                entry.worker_id for entry in route
            )

    for model_instance_id in model_instance_ids:
        model_rows = per_model_rows[model_instance_id]
        model_total = len(model_rows)
        model_latencies = [float(row["latency_sec"]) for row in model_rows]
        model_correct = sum(int(row["correct"]) for row in model_rows)
        model_stats = compute_latency_stats(model_latencies)

        prefix = f"{model_instance_id}_"
        results[f"{prefix}num_samples"] = int(model_total)
        results[f"{prefix}accuracy"] = compute_accuracy(model_correct, model_total)
        results[f"{prefix}throughput_samples_per_sec"] = compute_throughput(
            model_total,
            total_inference_time_sec,
        )
        results[f"{prefix}avg_latency_sec"] = float(model_stats["avg_latency_sec"])
        results[f"{prefix}p95_latency_sec"] = float(model_stats["p95_latency_sec"])
        results[f"{prefix}p99_latency_sec"] = float(model_stats["p99_latency_sec"])

        exit_summary = summarize_exit_counts(
            per_model_exit_counts[model_instance_id],
            model_total,
        )
        for key, value in exit_summary.items():
            results[f"{prefix}{key}"] = value

        for worker_id, compute_total in per_model_worker_compute[
            model_instance_id
        ].items():
            results[f"{prefix}{worker_id}_compute_time_total_sec"] = float(
                compute_total
            )

    worker_carbon_total = 0.0
    worker_energy_total = 0.0

    for worker_cfg in worker_cfgs:
        worker_id = str(worker_cfg["worker_id"])
        compute_total = float(worker_compute_totals[worker_id])
        req_total = int(stage_request_totals[worker_id])
        resp_total = int(stage_response_totals[worker_id])
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
        results[f"{worker_id}_request_bytes_total"] = req_total
        results[f"{worker_id}_response_bytes_total"] = resp_total
        results[f"{worker_id}_carbon_kg"] = worker_carbon_kg
        results[f"{worker_id}_energy_kWh"] = worker_energy_kwh

        if placement_assignments:
            assigned = placement_assignments.get(worker_id, [])
            results[f"{worker_id}_assigned_partitions"] = ",".join(
                f"{model_instance_id}:stage_{partition_id}"
                for model_instance_id, partition_id in assigned
            )
            for partition_id in range(num_workers):
                results[f"{worker_id}_num_stage_{partition_id}_partitions"] = sum(
                    1
                    for _, assigned_partition_id in assigned
                    if int(assigned_partition_id) == partition_id
                )

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

    per_sample_df = pd.DataFrame(flattened_rows)
    return results, per_sample_df


def main() -> None:
    args = parse_args()

    bundle = load_experiment_bundle(args.config)
    experiment_cfg = bundle["experiment_config"]
    dataset_cfg = bundle["dataset_config"]
    model_cfg = bundle["model_config"]
    system_cfg = bundle["system_config"]
    repo_root = bundle["repo_root"]

    runtime_cfg = experiment_cfg.get("runtime", {})
    output_dir = resolve_path(experiment_cfg["output"]["dir"], repo_root)
    data_dir = resolve_path(dataset_cfg["root"], repo_root)
    if output_dir is None or data_dir is None:
        raise ValueError("Failed to resolve output or data directory")

    batch_size = int(runtime_cfg.get("batch_size", 1))
    max_samples_per_model = runtime_cfg.get("max_samples_per_model")
    max_samples_per_model = (
        int(max_samples_per_model) if max_samples_per_model is not None else None
    )

    summary, per_sample_df = evaluate_multi_model_distributed_ee(
        experiment_cfg=experiment_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        system_cfg=system_cfg,
        data_dir=str(data_dir),
        repo_root=repo_root,
        batch_size=batch_size,
        max_samples_per_model=max_samples_per_model,
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
