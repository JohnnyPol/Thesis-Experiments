from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.models.partitioning import build_partition_module
from src.distributed.runtime.worker_monitoring import WorkerEmissionsMonitor


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
    model_name: str | None
    exit_policy: str | None
    emissions_monitor: WorkerEmissionsMonitor

    @property
    def is_final_stage(self) -> bool:
        return self.next_worker_cfg is None


def build_worker_runtime(
    *,
    worker_id: str,
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    system_cfg: dict[str, Any],
    repo_root: str,
) -> WorkerRuntime:
    worker_cfg = find_worker_cfg(system_cfg, worker_id)

    partition_id = int(worker_cfg["partition_id"])
    num_partitions = len(system_cfg.get("workers", []))
    device = torch.device(worker_cfg.get("device", "cpu"))
    host = str(worker_cfg.get("host"))
    port = int(worker_cfg["port"])
    next_worker_id = worker_cfg.get("next_worker_id")
    model_name = model_cfg.get("name")

    exit_policy = None
    if isinstance(model_cfg.get("early_exit"), dict):
        exit_policy = model_cfg["early_exit"].get("policy")
    if exit_policy is None:
        exit_policy = model_cfg.get("exit_policy")

    partition_module = build_partition_module(
        partition_id=partition_id,
        num_partitions=num_partitions,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        repo_root=repo_root,
        device=device,
    )

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
        model_name=model_name,
        exit_policy=exit_policy,
        emissions_monitor=WorkerEmissionsMonitor(),
    )
