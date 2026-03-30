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


def run_two_stage_inference(
    image_tensor: torch.Tensor,
    sample_id: int,
    worker1_cfg: dict[str, Any],
    worker2_cfg: dict[str, Any],
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """
    Run a single sample through the 2-worker early-exit pipeline.

    Flow:
      master -> worker1(input image)
      if worker1 exits: stop
      else master -> worker2(intermediate activation)
    """
    request_id_stage1 = f"{sample_id}-stage1-{uuid.uuid4().hex}"
    request_stage1 = make_infer_request(
        request_id=request_id_stage1,
        sample_id=sample_id,
        stage_id=0,
        tensor_bytes=tensor_to_bytes(image_tensor),
        request_type=REQUEST_TYPE_INFER_INPUT,
    )

    response_stage1, req1_bytes, resp1_bytes = roundtrip_request(
        host=_worker_connect_host(worker1_cfg),
        port=int(worker1_cfg["port"]),
        request=request_stage1,
        timeout_sec=timeout_sec,
    )
    validate_response(response_stage1)

    if response_stage1["status"] == RESPONSE_STATUS_ERROR:
        raise RuntimeError(f"Worker1 error: {response_stage1.get('error_message', 'unknown error')}")

    logical_bytes = req1_bytes + resp1_bytes
    total_remote_compute_sec = float(response_stage1.get("compute_time_sec", 0.0))

    if response_stage1["status"] == RESPONSE_STATUS_EXITED:
        logits = bytes_to_tensor(response_stage1["logits_bytes"], device="cpu")
        return {
            "logits": logits,
            "exit_id": int(response_stage1["exit_id"]),
            "worker1_compute_time_sec": float(response_stage1.get("compute_time_sec", 0.0)),
            "worker2_compute_time_sec": 0.0,
            "remote_compute_time_sec": total_remote_compute_sec,
            "protocol_bytes": int(logical_bytes),
            "stage1_request_bytes": int(req1_bytes),
            "stage1_response_bytes": int(resp1_bytes),
            "stage2_request_bytes": 0,
            "stage2_response_bytes": 0,
        }

    if response_stage1["status"] != RESPONSE_STATUS_FORWARDED:
        raise RuntimeError(f"Unexpected worker1 response status: {response_stage1['status']}")

    activation = bytes_to_tensor(response_stage1["activation_bytes"], device="cpu")

    request_id_stage2 = f"{sample_id}-stage2-{uuid.uuid4().hex}"
    request_stage2 = make_infer_request(
        request_id=request_id_stage2,
        sample_id=sample_id,
        stage_id=1,
        tensor_bytes=tensor_to_bytes(activation),
        request_type=REQUEST_TYPE_INFER_ACTIVATION,
    )

    response_stage2, req2_bytes, resp2_bytes = roundtrip_request(
        host=_worker_connect_host(worker2_cfg),
        port=int(worker2_cfg["port"]),
        request=request_stage2,
        timeout_sec=timeout_sec,
    )
    validate_response(response_stage2)

    if response_stage2["status"] == RESPONSE_STATUS_ERROR:
        raise RuntimeError(f"Worker2 error: {response_stage2.get('error_message', 'unknown error')}")

    if response_stage2["status"] not in {RESPONSE_STATUS_EXITED, RESPONSE_STATUS_COMPLETED}:
        raise RuntimeError(f"Unexpected worker2 response status: {response_stage2['status']}")

    logits = bytes_to_tensor(response_stage2["logits_bytes"], device="cpu")
    logical_bytes += req2_bytes + resp2_bytes
    total_remote_compute_sec += float(response_stage2.get("compute_time_sec", 0.0))

    return {
        "logits": logits,
        "exit_id": int(response_stage2["exit_id"]),
        "worker1_compute_time_sec": float(response_stage1.get("compute_time_sec", 0.0)),
        "worker2_compute_time_sec": float(response_stage2.get("compute_time_sec", 0.0)),
        "remote_compute_time_sec": total_remote_compute_sec,
        "protocol_bytes": int(logical_bytes),
        "stage1_request_bytes": int(req1_bytes),
        "stage1_response_bytes": int(resp1_bytes),
        "stage2_request_bytes": int(req2_bytes),
        "stage2_response_bytes": int(resp2_bytes),
    }