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
from src.metrics.latency import (
    compute_latency_stats,
    compute_throughput,
    compute_total_inference_time,
)
from src.metrics.network import compute_network_delta, read_network_bytes
from src.metrics.utilization import compute_node_utilization
from src.models.blocks import ResidualBlock
from src.models.resnet_baseline import ResNet
from src.utils.config import load_experiment_bundle, resolve_path


def evaluate_baseline_single_node(
    model: torch.nn.Module,
    model_name: str,
    data_dir: str = "./data",
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    network_interface: str | None = None,
    warmup_samples: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate a baseline ResNet model on a single node.
    """
    model.eval()
    device = torch.device(device)
    test_loader = data_loader(data_dir=data_dir, batch_size=batch_size, test=True)

    correct = 0
    total = 0
    latencies = []
    per_sample_rows = []

    if warmup_samples > 0:
        with torch.no_grad():
            warmup_count = 0
            for images, _ in test_loader:
                images = images.to(device)
                _ = model(images)
                warmup_count += images.size(0)
                if warmup_count >= warmup_samples:
                    break

        test_loader = data_loader(data_dir=data_dir, batch_size=batch_size, test=True)

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
            preds = model(images)
            end = time.time()

            latency = end - start
            latencies.append(latency)

            predicted = preds.argmax(dim=1)
            correct, total = update_correct_total(predicted, labels, correct, total)

            per_sample_rows.append(
                {
                    "sample_index": sample_index,
                    "batch_size": int(labels.size(0)),
                    "latency_sec": float(latency),
                    "predicted_class": int(predicted[0].item()),
                    "true_class": int(labels[0].item()),
                    "correct": int((predicted == labels).sum().item()),
                }
            )
            sample_index += 1

    experiment_end = time.time()
    tracker.stop()
    net_after = read_network_bytes(interface=network_interface)

    total_inference_time_sec = compute_total_inference_time(
        experiment_start, experiment_end
    )
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
        "mode": "single_node_baseline",
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
        "exit_0_count": None,
        "exit_1_count": None,
        "exit_2_count": None,
        "exit_3_count": None,
    }
    results.update(latency_stats)

    per_sample_df = pd.DataFrame(per_sample_rows)
    return results, per_sample_df


def build_resnet18_baseline(
    weights_path: str | None, device: torch.device | str
) -> torch.nn.Module:
    """
    Build and optionally load a baseline ResNet-18 model.
    """
    device = torch.device(device)
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    return model


def save_results(
    output_dir: str, summary: dict, per_sample_df: pd.DataFrame, config_bundle: dict
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
    parser = argparse.ArgumentParser(description="Run baseline single-node inference.")
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

    output_dir = resolve_path(experiment_cfg["output"]["dir"], repo_root) or "./results/exp1_single_model/01_single_node_baseline/run_001"
    data_dir = resolve_path(dataset_cfg["root"], repo_root) or "./data"
    weights_path = resolve_path(model_cfg["weights"]["path"], repo_root)

    batch_size = experiment_cfg.get("runtime", {}).get(
        "batch_size",
        dataset_cfg.get("loader", {}).get("batch_size", 1),
    )
    warmup_samples = experiment_cfg.get("runtime", {}).get("warmup_samples", 0)
    device = system_cfg.get("runtime", {}).get("device", "cpu")
    network_interface = system_cfg.get("monitoring", {}).get("network_interface", None)
    model_name = model_cfg.get("name", "resnet18_baseline")

    model = build_resnet18_baseline(
        weights_path=weights_path,
        device=device,
    )

    summary, per_sample_df = evaluate_baseline_single_node(
        model=model,
        model_name=model_name,
        data_dir=data_dir,
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

    save_results(output_dir, summary, per_sample_df, bundle)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
