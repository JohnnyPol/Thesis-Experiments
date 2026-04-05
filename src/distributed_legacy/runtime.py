import torch

from src.distributed_legacy.monitoring import WorkerEmissionsMonitor
from src.distributed_legacy.partitioning import build_partition_module


class WorkerRuntime(object):
    def __init__(
        self,
        worker_id,
        partition_id,
        num_partitions,
        device,
        host,
        port,
        next_worker_id,
        worker_cfg,
        partition_module,
        model_name,
        exit_policy,
        emissions_monitor,
    ):
        self.worker_id = worker_id
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.device = device
        self.host = host
        self.port = port
        self.next_worker_id = next_worker_id
        self.worker_cfg = worker_cfg
        self.partition_module = partition_module
        self.model_name = model_name
        self.exit_policy = exit_policy
        self.emissions_monitor = emissions_monitor


def _resolve_device(device_name):
    normalized = str(device_name or "cpu").strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    return torch.device(normalized)


def find_worker_cfg(system_cfg, worker_id):
    for worker_cfg in system_cfg.get("workers", []):
        if str(worker_cfg.get("worker_id")) == str(worker_id):
            return worker_cfg
    raise ValueError("Worker '{0}' not found in system config".format(worker_id))


def build_worker_runtime(worker_id, dataset_cfg, model_cfg, system_cfg, repo_root):
    worker_cfg = find_worker_cfg(system_cfg, worker_id)

    partition_id = int(worker_cfg["partition_id"])
    num_partitions = len(system_cfg.get("workers", []))
    device = _resolve_device(worker_cfg.get("device", "cpu"))
    host = str(worker_cfg.get("host"))
    port = int(worker_cfg["port"])
    next_worker_id = worker_cfg.get("next_worker_id")
    model_name = model_cfg.get("name")

    if next_worker_id is not None:
        raise ValueError(
            "Legacy Jetson runtime only supports the final worker. "
            "worker_id={0} has next_worker_id={1}".format(worker_id, next_worker_id)
        )

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

    return WorkerRuntime(
        worker_id=str(worker_id),
        partition_id=partition_id,
        num_partitions=num_partitions,
        device=device,
        host=host,
        port=port,
        next_worker_id=None,
        worker_cfg=worker_cfg,
        partition_module=partition_module,
        model_name=model_name,
        exit_policy=exit_policy,
        emissions_monitor=WorkerEmissionsMonitor(),
    )
