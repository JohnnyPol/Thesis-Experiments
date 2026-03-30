from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from codecarbon import EmissionsTracker

from src.data.loaders import data_loader
from src.metrics.accuracy import compute_accuracy, update_correct_total
from src.metrics.exits import initialize_exit_counts, summarize_exit_counts, update_exit_counts
from src.metrics.latency import (
    compute_latency_stats,
    compute_throughput,
    compute_total_inference_time,
)
from src.metrics.network import compute_network_delta, read_network_bytes
from src.metrics.utilization import compute_node_utilization
from src.models.blocks import ResidualBlock
from src.models.resnet_baseline import ResNet
from src.models.resnet_ee import ResNetEE18
from src.utils.config import load_experiment_bundle, resolve_path


def is_early_exit_model(model_cfg: dict) -> bool:
    """
    Detect whether the loaded model config refers to an early-exit model.

    This is intentionally flexible, so it can work with slightly different YAML schemas.
    """
    name = str(model_cfg.get("name", "")).lower()
    architecture = str(model_cfg.get("architecture", "")).lower()
    model_type = str(model_cfg.get("type", "")).lower()
    variant = str(model_cfg.get("variant", "")).lower()

    flags = [
        "ee" in name,
        "early_exit" in name,
        "ee" in architecture,
        "early_exit" in architecture,
        model_type in {"ee", "early_exit"},
        variant in {"ee", "early_exit", "entropy"},
        bool(model_cfg.get("early_exit", False)),
    ]
    return any(flags)


def extract_num_classes(dataset_cfg: dict, model_cfg: dict) -> int:
    """
    Prefer dataset num_classes, fallback to model config, default to 10.
    """
    return int(
        dataset_cfg.get(
            "num_classes",
            model_cfg.get("num_classes", 10),
        )
    )


def extract_entropy_threshold(model_cfg: dict) -> float:
    """
    Try a few common locations for the entropy threshold in the YAML config.
    """
    if "confidence_threshold" in model_cfg:
        return float(model_cfg["confidence_threshold"])

    if "exit_policy" in model_cfg and isinstance(model_cfg["exit_policy"], dict):
        if "confidence_threshold" in model_cfg["exit_policy"]:
            return float(model_cfg["exit_policy"]["confidence_threshold"])
        if "entropy_threshold" in model_cfg["exit_policy"]:
            return float(model_cfg["exit_policy"]["entropy_threshold"])

    if "early_exit" in model_cfg and isinstance(model_cfg["early_exit"], dict):
        if "confidence_threshold" in model_cfg["early_exit"]:
            return float(model_cfg["early_exit"]["confidence_threshold"])
        if "entropy_threshold" in model_cfg["early_exit"]:
            return float(model_cfg["early_exit"]["entropy_threshold"])

    return 0.9


def build_model_from_config(
    model_cfg: dict,
    dataset_cfg: dict,
    weights_path: str | None,
    device: torch.device | str,
) -> tuple[torch.nn.Module, str]:
    """
    Build either baseline ResNet-18 or EE ResNet-18 from config.
    """
    device = torch.device(device)
    model_name = model_cfg.get("name", "model")

    num_classes = extract_num_classes(dataset_cfg, model_cfg)

    if is_early_exit_model(model_cfg):
        confidence_threshold = extract_entropy_threshold(model_cfg)
        model = ResNetEE18(
            ResidualBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
        ).to(device)
    else:
        model = ResNet(
            ResidualBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
        ).to(device)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    return model, model_name


def evaluate_single_node(
    model: torch.nn.Module,
    model_name: str,
    dataset_config: dict,
    is_ee: bool,
    data_dir: str = "./data",
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    network_interface: str | None = None,
    warmup_samples: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate either a baseline or an early-exit model on a single node.
    """
    model.eval()
    device = torch.device(device)

    if is_ee and batch_size != 1:
        raise ValueError(
            "Early-exit inference currently requires batch_size=1 because "
            "ResNetEE._confident_enough() uses entropy.item() on a single sample."
        )

    num_workers = dataset_config.get("loader", {}).get("num_workers", 0)

    test_loader = data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        test=True,
        num_workers=num_workers,
        dataset_config=dataset_config,
    )

    correct = 0
    total = 0
    latencies: list[float] = []
    per_sample_rows: list[dict] = []

    exit_counts = initialize_exit_counts(4) if is_ee else None

    if warmup_samples > 0:
        with torch.no_grad():
            warmup_count = 0
            for images, _ in test_loader:
                images = images.to(device)
                _ = model(images)
                warmup_count += images.size(0)
                if warmup_count >= warmup_samples:
                    break

        test_loader = data_loader(
            data_dir=data_dir,
            batch_size=batch_size,
            test=True,
            num_workers=num_workers,
            dataset_config=dataset_config,
        )

    net_before = read_network_bytes(interface=network_interface)

    tracker = EmissionsTracker(measure_power_secs=1)
    tracker.start()
    experiment_start = time.time()

    with torch.no_grad():
        sample_index = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            start = time.time()
            outputs = model(images)
            end = time.time()

            latency = end - start
            latencies.append(latency)

            if is_ee:
                logits, exit_id = outputs
                exit_id = int(exit_id)
                assert exit_counts is not None
                update_exit_counts(exit_counts, exit_id)
            else:
                logits = outputs
                exit_id = None

            predicted = logits.argmax(dim=1)
            correct, total = update_correct_total(predicted, labels, correct, total)

            row = {
                "sample_index": sample_index,
                "batch_size": int(labels.size(0)),
                "latency_sec": float(latency),
                "predicted_class": int(predicted[0].item()),
                "true_class": int(labels[0].item()),
                "correct": int((predicted == labels).sum().item()),
                "exit_id": exit_id,
            }
            per_sample_rows.append(row)
            sample_index += 1

    experiment_end = time.time()
    tracker.stop()
    net_after = read_network_bytes(interface=network_interface)

    total_inference_time_sec = compute_total_inference_time(experiment_start, experiment_end)
    latency_stats = compute_latency_stats(latencies)
    throughput = compute_throughput(total, total_inference_time_sec)
    node_utilization = compute_node_utilization(
        latency_stats["busy_time_sec"],
        total_inference_time_sec,
    )
    accuracy = compute_accuracy(correct, total)
    network_stats = compute_network_delta(net_before, net_after)

    emissions_data = tracker._prepare_emissions_data()
    carbon_kg = emissions_data.emissions
    energy_kwh = emissions_data.energy_consumed

    results = {
        "model_name": model_name,
        "mode": "single_node_early_exit" if is_ee else "single_node_baseline",
        "accuracy": accuracy,
        "num_correct": int(correct),
        "num_samples": int(total),
        "total_inference_time_sec": float(total_inference_time_sec),
        "throughput_samples_per_sec": float(throughput),
        "node_utilization": float(node_utilization),
        "carbon_kg": float(carbon_kg) if carbon_kg is not None else None,
        "energy_kWh": float(energy_kwh) if energy_kwh is not None else None,
        "network_rx_bytes": int(network_stats["rx_bytes"]),
        "network_tx_bytes": int(network_stats["tx_bytes"]),
        "network_total_bytes": int(network_stats["total_bytes"]),
    }
    results.update(latency_stats)

    if is_ee and exit_counts is not None:
        results.update(summarize_exit_counts(exit_counts, total))
    else:
        results["exit_0_count"] = None
        results["exit_0_ratio"] = None
        results["exit_1_count"] = None
        results["exit_1_ratio"] = None
        results["exit_2_count"] = None
        results["exit_2_ratio"] = None
        results["exit_3_count"] = None
        results["exit_3_ratio"] = None

    per_sample_df = pd.DataFrame(per_sample_rows)
    return results, per_sample_df


def save_results(
    output_dir: str,
    summary: dict,
    per_sample_df: pd.DataFrame,
    config_bundle: dict,
) -> None:
    """
    Save experiment outputs to disk.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_path = out_path / "metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    per_sample_path = out_path / "latencies.csv"
    per_sample_df.to_csv(per_sample_path, index=False)

    config_dump_path = out_path / "resolved_config.json"
    with open(config_dump_path, "w", encoding="utf-8") as f:
        json.dump(config_bundle, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-node inference for baseline or early-exit models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    bundle = load_experiment_bundle(args.config)
    experiment_cfg = bundle["experiment_config"]
    dataset_cfg = bundle["dataset_config"]
    model_cfg = bundle["model_config"]
    system_cfg = bundle["system_config"]
    repo_root = bundle["repo_root"]

    output_dir = resolve_path(experiment_cfg["output"]["dir"], repo_root)
    data_dir = resolve_path(dataset_cfg["root"], repo_root)

    weights_path = None
    if "weights" in model_cfg and isinstance(model_cfg["weights"], dict):
        weights_path = resolve_path(model_cfg["weights"].get("path"), repo_root)

    batch_size = experiment_cfg.get("runtime", {}).get(
        "batch_size",
        dataset_cfg.get("loader", {}).get("batch_size", 1),
    )
    warmup_samples = experiment_cfg.get("runtime", {}).get("warmup_samples", 0)
    device = system_cfg.get("runtime", {}).get("device", "cpu")
    network_interface = system_cfg.get("monitoring", {}).get("network_interface", None)

    is_ee = is_early_exit_model(model_cfg)

    model, model_name = build_model_from_config(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        weights_path=weights_path,
        device=device,
    )

    summary, per_sample_df = evaluate_single_node(
        model=model,
        model_name=model_name,
        dataset_config=dataset_cfg,
        is_ee=is_ee,
        data_dir=data_dir,  # type: ignore
        batch_size=batch_size,
        device=device,
        network_interface=network_interface,
        warmup_samples=warmup_samples,
    )

    summary["experiment_id"] = experiment_cfg.get("experiment", {}).get("id")
    summary["experiment_name"] = experiment_cfg.get("experiment", {}).get("name")
    summary["dataset_name"] = dataset_cfg.get("name")
    summary["system_name"] = system_cfg.get("system_name")
    summary["weights_path"] = weights_path
    summary["data_dir"] = data_dir
    summary["output_dir"] = output_dir
    summary["is_early_exit_model"] = is_ee

    save_results(output_dir, summary, per_sample_df, bundle)  # type: ignore

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()