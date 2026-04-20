from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.models.partitioning import build_partition_module
from src.distributed.runtime.worker_monitoring import WorkerEmissionsMonitor


def _resolve_device(device_name: str | None) -> torch.device:
    normalized = str(device_name or "cpu").strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    return torch.device(normalized)


def find_worker_cfg(system_cfg: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker_cfg in system_cfg.get("workers", []):
        if str(worker_cfg.get("worker_id")) == str(worker_id):
            return worker_cfg
    raise ValueError(f"Worker '{worker_id}' not found in system config")


def resolve_next_worker_cfg(
    system_cfg: dict[str, Any],
    worker_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    next_worker_id = worker_cfg.get("next_worker_id")
    if next_worker_id is None:
        return None
    return find_worker_cfg(system_cfg, str(next_worker_id))


@dataclass
class WorkerRuntime:
    worker_id: str
    partition_id: int
    num_partitions: int
    device: torch.device
    host: str
    port: int
    next_worker_id: str | None
    worker_cfg: dict[str, Any]
    next_worker_cfg: dict[str, Any] | None
    partition_module: torch.nn.Module
    partition_modules: dict[str, torch.nn.Module]
    model_instance_ids: list[str]
    model_name: str | None
    exit_policy: str | None
    emissions_monitor: WorkerEmissionsMonitor

    @property
    def is_final_stage(self) -> bool:
        return self.next_worker_cfg is None

    def get_partition_module(self, model_instance_id: str | None) -> torch.nn.Module:
        resolved_id = str(model_instance_id or "model_0")
        if resolved_id not in self.partition_modules:
            raise ValueError(
                f"Worker {self.worker_id} does not have a partition for "
                f"model_instance_id='{resolved_id}'"
            )
        return self.partition_modules[resolved_id]


def resolve_model_instance_ids(
    *,
    experiment_cfg: dict[str, Any] | None,
    system_cfg: dict[str, Any],
) -> list[str]:
    runtime_cfg = (experiment_cfg or {}).get("runtime", {})
    raw_count = runtime_cfg.get("model_instance_count", 1)
    num_workers = len(system_cfg.get("workers", []))

    if raw_count == "auto":
        count = max(num_workers - 1, 1)
    else:
        count = int(raw_count)

    if count < 1:
        raise ValueError("model_instance_count must be at least 1")

    return [f"model_{idx}" for idx in range(count)]


def build_worker_runtime(
    *,
    worker_id: str,
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    system_cfg: dict[str, Any],
    repo_root: str,
    experiment_cfg: dict[str, Any] | None = None,
) -> WorkerRuntime:
    worker_cfg = find_worker_cfg(system_cfg, worker_id)

    partition_id = int(worker_cfg["partition_id"])
    num_partitions = len(system_cfg.get("workers", []))
    device = _resolve_device(worker_cfg.get("device", "cpu"))
    host = str(worker_cfg.get("host"))
    port = int(worker_cfg["port"])
    next_worker_id = worker_cfg.get("next_worker_id")
    model_name = model_cfg.get("name")

    exit_policy = None
    if isinstance(model_cfg.get("early_exit"), dict):
        exit_policy = model_cfg["early_exit"].get("policy")
    if exit_policy is None:
        exit_policy = model_cfg.get("exit_policy")

    model_instance_ids = resolve_model_instance_ids(
        experiment_cfg=experiment_cfg,
        system_cfg=system_cfg,
    )
    partition_modules = {
        model_instance_id: build_partition_module(
            partition_id=partition_id,
            num_partitions=num_partitions,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            repo_root=repo_root,
            device=device,
        )
        for model_instance_id in model_instance_ids
    }
    partition_module = partition_modules[model_instance_ids[0]]

    next_worker_cfg = resolve_next_worker_cfg(system_cfg, worker_cfg)

    return WorkerRuntime(
        worker_id=str(worker_id),
        partition_id=partition_id,
        num_partitions=num_partitions,
        device=device,
        host=host,
        port=port,
        next_worker_id=str(next_worker_id) if next_worker_id is not None else None,
        worker_cfg=worker_cfg,
        next_worker_cfg=next_worker_cfg,
        partition_module=partition_module,
        partition_modules=partition_modules,
        model_instance_ids=model_instance_ids,
        model_name=model_name,
        exit_policy=exit_policy,
        emissions_monitor=WorkerEmissionsMonitor(),
    )
