from __future__ import annotations

import uuid
from typing import Any

import torch

from src.distributed.api.schemas import InferenceRequestMetadata
from src.distributed.client.fastapi_client import infer_remote
from src.distributed.protocol.constants import REQUEST_KIND_INPUT
from src.distributed.protocol.tensor_codec import tensor_to_bytes


def run_chained_inference(
    image_tensor: torch.Tensor,
    sample_id: int,
    entry_worker_cfg: dict[str, Any],
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """
    Send one sample to the entry worker. Downstream forwarding is handled
    internally by workers.

    Returns:
        {
            "predicted_class": int,
            "confidence": float | None,
            "exit_id": int,
            "protocol_bytes": int,
            "remote_compute_time_sec": float,
            "worker_compute_times": {worker_id: float},
            "stage_request_bytes": {worker_id: int},
            "stage_response_bytes": {worker_id: int},
            "path": [worker_ids...],
        }
    """
    entry_worker_id = str(entry_worker_cfg["worker_id"])

    request_id = f"{sample_id}-{uuid.uuid4().hex}"
    trace_id = request_id

    tensor_bytes, tensor_shape, tensor_dtype = tensor_to_bytes(image_tensor)

    metadata = InferenceRequestMetadata(
        request_id=request_id,
        sample_id=int(sample_id),
        trace_id=trace_id,
        request_kind=REQUEST_KIND_INPUT,
        stage_id=int(entry_worker_cfg.get("partition_id", 0)),
        origin_node="master",
        current_node=entry_worker_id,
        next_node=entry_worker_cfg.get("next_worker_id"),
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        tensor_layout="NCHW",
    )

    terminal, req_bytes, resp_bytes = infer_remote(
        worker_cfg=entry_worker_cfg,
        metadata=metadata,
        tensor_bytes=tensor_bytes,
        timeout_sec=timeout_sec,
    )

    worker_compute_times: dict[str, float] = {}
    stage_request_bytes: dict[str, int] = {}
    stage_response_bytes: dict[str, int] = {}

    for metric in terminal.stage_metrics:
        worker_id = str(metric.worker_id)
        worker_compute_times[worker_id] = float(metric.compute_time_sec)
        stage_request_bytes[worker_id] = int(metric.request_bytes)
        stage_response_bytes[worker_id] = int(metric.response_bytes)

    protocol_bytes = int(terminal.total_protocol_bytes)
    if protocol_bytes <= 0:
        protocol_bytes = int(req_bytes + resp_bytes)

    remote_compute_time_sec = float(terminal.total_remote_compute_time_sec)
    if remote_compute_time_sec <= 0.0:
        remote_compute_time_sec = float(
            sum(metric.compute_time_sec for metric in terminal.stage_metrics)
        )

    return {
        "predicted_class": int(terminal.predicted_class)
        if terminal.predicted_class is not None
        else -1,
        "confidence": terminal.confidence,
        "exit_id": int(terminal.exit_id),
        "protocol_bytes": protocol_bytes,
        "remote_compute_time_sec": remote_compute_time_sec,
        "worker_compute_times": worker_compute_times,
        "stage_request_bytes": stage_request_bytes,
        "stage_response_bytes": stage_response_bytes,
        "path": list(terminal.path),
    }