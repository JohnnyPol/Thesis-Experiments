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
class RouteEntry:
    worker_id: str
    partition_id: int


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
    partition_modules: dict[tuple[str, int], torch.nn.Module]
    model_instance_ids: list[str]
    placement_enabled: bool
    placement_assignments: dict[str, list[tuple[str, int]]]
    placement_routes: dict[str, list[RouteEntry]]
    model_name: str | None
    dataset_name: str | None
    exit_policy: str | None
    emissions_monitor: WorkerEmissionsMonitor

    @property
    def is_final_stage(self) -> bool:
        return self.next_worker_cfg is None

    def get_partition_module(
        self,
        model_instance_id: str | None,
        partition_id: int | None = None,
    ) -> torch.nn.Module:
        resolved_id = str(model_instance_id or "model_0")
        resolved_partition_id = (
            int(partition_id) if partition_id is not None else self.partition_id
        )
        key = (resolved_id, resolved_partition_id)
        if key not in self.partition_modules:
            raise ValueError(
                f"Worker {self.worker_id} does not have a partition for "
                f"model_instance_id='{resolved_id}', partition_id={resolved_partition_id}"
            )
        return self.partition_modules[key]

    def resolve_current_partition_id(self, metadata_stage_id: int) -> int:
        if self.placement_enabled:
            return int(metadata_stage_id)
        return int(self.partition_id)

    def resolve_next_route_entry(
        self,
        model_instance_id: str,
        current_partition_id: int,
    ) -> RouteEntry | None:
        if not self.placement_enabled:
            if self.next_worker_cfg is None:
                return None
            return RouteEntry(
                worker_id=str(self.next_worker_cfg["worker_id"]),
                partition_id=int(current_partition_id) + 1,
            )

        route = self.placement_routes.get(str(model_instance_id))
        if not route:
            raise ValueError(f"No placement route configured for {model_instance_id}")

        for index, entry in enumerate(route):
            if int(entry.partition_id) != int(current_partition_id):
                continue
            if str(entry.worker_id) != self.worker_id:
                raise ValueError(
                    f"Route for {model_instance_id} stage {current_partition_id} "
                    f"expects worker {entry.worker_id}, got {self.worker_id}"
                )
            if index + 1 >= len(route):
                return None
            return route[index + 1]

        raise ValueError(
            f"Worker {self.worker_id} is not on the route for "
            f"{model_instance_id} stage {current_partition_id}"
        )

    def peek_next_route_entry(
        self,
        model_instance_id: str,
        current_partition_id: int,
    ) -> RouteEntry | None:
        if not self.placement_enabled:
            next_partition_id = int(current_partition_id) + 1
            for worker_cfg in self._all_worker_cfgs:
                if int(worker_cfg.get("partition_id", -1)) == next_partition_id:
                    return RouteEntry(
                        worker_id=str(worker_cfg["worker_id"]),
                        partition_id=next_partition_id,
                    )
            return None

        route = self.placement_routes.get(str(model_instance_id), [])
        for index, entry in enumerate(route):
            if int(entry.partition_id) == int(current_partition_id):
                if index + 1 >= len(route):
                    return None
                return route[index + 1]
        return None

    def get_worker_cfg(self, worker_id: str) -> dict[str, Any]:
        return find_worker_cfg({"workers": self._all_worker_cfgs}, worker_id)

    @property
    def _all_worker_cfgs(self) -> list[dict[str, Any]]:
        return self.worker_cfg.get("_all_worker_cfgs", [])


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


def _parse_route_entries(raw_entries: list[dict[str, Any]]) -> list[RouteEntry]:
    return [
        RouteEntry(
            worker_id=str(entry["worker_id"]),
            partition_id=int(entry["partition_id"]),
        )
        for entry in raw_entries
    ]


def resolve_placement_routes(
    experiment_cfg: dict[str, Any] | None,
) -> dict[str, list[RouteEntry]]:
    placement_cfg = (experiment_cfg or {}).get("placement")
    if not isinstance(placement_cfg, dict):
        return {}
    routes_cfg = placement_cfg.get("routes")
    if not isinstance(routes_cfg, dict):
        return {}
    return {
        str(model_instance_id): _parse_route_entries(route_entries)
        for model_instance_id, route_entries in routes_cfg.items()
    }


def resolve_worker_assignments(
    experiment_cfg: dict[str, Any] | None,
) -> dict[str, list[tuple[str, int]]]:
    placement_cfg = (experiment_cfg or {}).get("placement")
    if not isinstance(placement_cfg, dict):
        return {}
    assignments_cfg = placement_cfg.get("assignments")
    if not isinstance(assignments_cfg, dict):
        return {}

    assignments: dict[str, list[tuple[str, int]]] = {}
    for worker_id, entries in assignments_cfg.items():
        assignments[str(worker_id)] = [
            (str(entry["model_instance_id"]), int(entry["partition_id"]))
            for entry in entries
        ]
    return assignments


def validate_placement_config(
    *,
    experiment_cfg: dict[str, Any],
    system_cfg: dict[str, Any],
) -> None:
    assignments = resolve_worker_assignments(experiment_cfg)
    routes = resolve_placement_routes(experiment_cfg)
    if not assignments and not routes:
        return
    if not assignments or not routes:
        raise ValueError("Experiment placement must define both assignments and routes")

    worker_ids = {str(worker["worker_id"]) for worker in system_cfg.get("workers", [])}
    assigned_pair_workers = {
        (model_instance_id, partition_id): worker_id
        for worker_id, entries in assignments.items()
        for model_instance_id, partition_id in entries
    }

    for worker_id in assignments:
        if worker_id not in worker_ids:
            raise ValueError(f"Placement assignment references unknown worker {worker_id}")

    for model_instance_id, route in routes.items():
        partition_ids = [entry.partition_id for entry in route]
        if partition_ids != sorted(partition_ids):
            raise ValueError(f"Route for {model_instance_id} must be ordered by partition_id")
        if partition_ids != list(range(len(route))):
            raise ValueError(
                f"Route for {model_instance_id} must contain contiguous partition ids "
                f"starting at 0, got {partition_ids}"
            )
        for entry in route:
            if entry.worker_id not in worker_ids:
                raise ValueError(
                    f"Route for {model_instance_id} references unknown worker {entry.worker_id}"
                )
            pair = (model_instance_id, int(entry.partition_id))
            if pair not in assigned_pair_workers:
                raise ValueError(f"Route entry {pair} has no matching worker assignment")
            if assigned_pair_workers[pair] != entry.worker_id:
                raise ValueError(
                    f"Route entry {pair} points to {entry.worker_id}, but assignment "
                    f"places it on {assigned_pair_workers[pair]}"
                )

    stage_zero_counts: dict[str, int] = {}
    for worker_id, entries in assignments.items():
        stage_zero_counts[worker_id] = sum(
            1 for _, partition_id in entries if int(partition_id) == 0
        )
        if stage_zero_counts[worker_id] > 1:
            raise ValueError(
                f"Worker {worker_id} has more than one first-stage partition"
            )


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
    worker_cfg = {**worker_cfg, "_all_worker_cfgs": list(system_cfg.get("workers", []))}

    partition_id = int(worker_cfg["partition_id"])
    num_partitions = len(system_cfg.get("workers", []))
    device = _resolve_device(worker_cfg.get("device", "cpu"))
    host = str(worker_cfg.get("host"))
    port = int(worker_cfg["port"])
    next_worker_id = worker_cfg.get("next_worker_id")
    model_name = model_cfg.get("name")
    dataset_name = dataset_cfg.get("name")

    exit_policy = None
    if isinstance(model_cfg.get("early_exit"), dict):
        exit_policy = model_cfg["early_exit"].get("policy")
    if exit_policy is None:
        exit_policy = model_cfg.get("exit_policy")

    model_instance_ids = resolve_model_instance_ids(
        experiment_cfg=experiment_cfg,
        system_cfg=system_cfg,
    )
    validate_placement_config(
        experiment_cfg=experiment_cfg or {},
        system_cfg=system_cfg,
    )
    placement_routes = resolve_placement_routes(experiment_cfg)
    placement_assignments_raw = resolve_worker_assignments(experiment_cfg)
    placement_enabled = bool(placement_routes and placement_assignments_raw)

    if placement_enabled:
        assigned_pairs = placement_assignments_raw.get(str(worker_id), [])
    else:
        assigned_pairs = [
            (model_instance_id, partition_id) for model_instance_id in model_instance_ids
        ]

    if not assigned_pairs:
        raise ValueError(f"No partitions assigned to worker {worker_id}")

    partition_modules = {
        (model_instance_id, assigned_partition_id): build_partition_module(
            partition_id=assigned_partition_id,
            num_partitions=num_partitions,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            repo_root=repo_root,
            device=device,
        )
        for model_instance_id, assigned_partition_id in assigned_pairs
    }
    partition_module = next(iter(partition_modules.values()))

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
        placement_enabled=placement_enabled,
        placement_assignments=placement_assignments_raw,
        placement_routes=placement_routes,
        model_name=model_name,
        dataset_name=dataset_name,
        exit_policy=exit_policy,
        emissions_monitor=WorkerEmissionsMonitor(),
    )
