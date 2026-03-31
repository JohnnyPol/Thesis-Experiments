from __future__ import annotations

import uuid
from typing import Any

import torch

from src.distributed.rpc_messages import (
    REQUEST_TYPE_INFER_ACTIVATION,
    REQUEST_TYPE_INFER_INPUT,
    RESPONSE_STATUS_COMPLETED,
    RESPONSE_STATUS_ERROR,
    RESPONSE_STATUS_EXITED,
    RESPONSE_STATUS_FORWARDED,
    make_infer_request,
    validate_response,
)
from src.distributed.serialization import (
    bytes_to_tensor,
    roundtrip_request,
    tensor_to_bytes,
)


def _worker_connect_host(worker_cfg: dict[str, Any]) -> str:
    return str(worker_cfg.get("connect_host", worker_cfg["host"]))


def run_multi_stage_inference(
    image_tensor: torch.Tensor,
    sample_id: int,
    worker_cfgs: list[dict[str, Any]],
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """
    Run a single sample through an ordered multi-stage early-exit pipeline.

    Stage 0 receives raw input image.
    Subsequent stages receive forwarded activation tensors.

    Returns:
        {
            "logits": tensor,
            "exit_id": int,
            "protocol_bytes": int,
            "remote_compute_time_sec": float,
            "worker_compute_times": {worker_id: float},
            "stage_request_bytes": {worker_id: int},
            "stage_response_bytes": {worker_id: int},
        }
    """
    if len(worker_cfgs) == 0:
        raise ValueError("worker_cfgs must contain at least one worker")

    current_tensor = image_tensor
    current_request_type = REQUEST_TYPE_INFER_INPUT

    protocol_bytes_total = 0
    remote_compute_time_total = 0.0
    worker_compute_times: dict[str, float] = {}
    stage_request_bytes: dict[str, int] = {}
    stage_response_bytes: dict[str, int] = {}

    for stage_index, worker_cfg in enumerate(worker_cfgs):
        worker_id = str(worker_cfg["worker_id"])

        request_id = f"{sample_id}-stage{stage_index}-{uuid.uuid4().hex}"
        request = make_infer_request(
            request_id=request_id,
            sample_id=sample_id,
            stage_id=stage_index,
            tensor_bytes=tensor_to_bytes(current_tensor),
            request_type=current_request_type,
        )

        response, req_bytes, resp_bytes = roundtrip_request(
            host=_worker_connect_host(worker_cfg),
            port=int(worker_cfg["port"]),
            request=request,
            timeout_sec=timeout_sec,
        )
        validate_response(response)

        if response["status"] == RESPONSE_STATUS_ERROR:
            raise RuntimeError(
                f"{worker_id} error: {response.get('error_message', 'unknown error')}"
            )

        protocol_bytes_total += req_bytes + resp_bytes
        remote_compute_time_total += float(response.get("compute_time_sec", 0.0))
        worker_compute_times[worker_id] = float(response.get("compute_time_sec", 0.0))
        stage_request_bytes[worker_id] = int(req_bytes)
        stage_response_bytes[worker_id] = int(resp_bytes)

        if response["status"] in {RESPONSE_STATUS_EXITED, RESPONSE_STATUS_COMPLETED}:
            logits = bytes_to_tensor(response["logits_bytes"], device="cpu")
            return {
                "logits": logits,
                "exit_id": int(response["exit_id"]),
                "protocol_bytes": int(protocol_bytes_total),
                "remote_compute_time_sec": float(remote_compute_time_total),
                "worker_compute_times": worker_compute_times,
                "stage_request_bytes": stage_request_bytes,
                "stage_response_bytes": stage_response_bytes,
            }

        if response["status"] != RESPONSE_STATUS_FORWARDED:
            raise RuntimeError(
                f"Unexpected response status from {worker_id}: {response['status']}"
            )

        current_tensor = bytes_to_tensor(response["activation_bytes"], device="cpu")
        current_request_type = REQUEST_TYPE_INFER_ACTIVATION

    raise RuntimeError("Pipeline ended without a terminal response")