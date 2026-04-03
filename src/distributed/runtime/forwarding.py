from __future__ import annotations

import time
from typing import Any

import torch

from src.distributed.api.schemas import (
    InferenceRequestMetadata,
    StageMetric,
    TerminalInferenceResponse,
)
from src.distributed.client.fastapi_client import infer_remote
from src.distributed.protocol.constants import (
    REQUEST_KIND_ACTIVATION,
    RESPONSE_STATUS_COMPLETED,
    RESPONSE_STATUS_EXITED,
)
from src.distributed.protocol.tensor_codec import (
    tensor_to_bytes,
    torch_dtype_to_str,
)
from src.distributed.runtime.worker_runtime import WorkerRuntime


def execute_or_forward(
    *,
    runtime: WorkerRuntime,
    metadata: InferenceRequestMetadata,
    tensor: torch.Tensor,
    inbound_request_bytes: int = 0,
) -> TerminalInferenceResponse:
    """
    Execute this worker's partition on the given tensor.

    If the partition exits or completes, return a terminal response.
    If it needs to forward, call the next worker and then enrich the downstream
    terminal response with this worker's stage metrics.
    """
    tensor_on_device = tensor.to(runtime.device)

    with torch.no_grad():
        output = runtime.partition_module(tensor_on_device)

    local_compute_time_sec = float(output.compute_time_sec)

    if output.status in {RESPONSE_STATUS_EXITED, RESPONSE_STATUS_COMPLETED}:
        logits = _extract_logits_cpu(output.logits)
        predicted_class, confidence = _compute_prediction_summary(logits)

        logits_shape = list(logits.shape)
        logits_dtype = torch_dtype_to_str(logits.dtype)

        local_response_bytes = _estimate_terminal_response_bytes(
            request_id=metadata.request_id,
            sample_id=metadata.sample_id,
            trace_id=metadata.trace_id,
            worker_id=runtime.worker_id,
            stage_id=runtime.partition_id,
            exit_id=int(output.exit_id),
            predicted_class=predicted_class,
            confidence=confidence,
            logits_shape=logits_shape,
            logits_dtype=logits_dtype,
        )

        local_stage_metric = StageMetric(
            worker_id=runtime.worker_id,
            stage_id=runtime.partition_id,
            compute_time_sec=local_compute_time_sec,
            request_bytes=int(inbound_request_bytes),
            response_bytes=int(local_response_bytes),
        )

        terminal = TerminalInferenceResponse(
            status=output.status,
            request_id=metadata.request_id,
            sample_id=metadata.sample_id,
            trace_id=metadata.trace_id,
            worker_id=runtime.worker_id,
            stage_id=runtime.partition_id,
            exit_id=int(output.exit_id),
            predicted_class=predicted_class,
            confidence=confidence,
            logits_shape=logits_shape,
            logits_dtype=logits_dtype,
            compute_time_sec=local_compute_time_sec,
            stage_metrics=[local_stage_metric],
            path=[runtime.worker_id],
            total_request_bytes=int(inbound_request_bytes),
            total_response_bytes=int(local_response_bytes),
            total_protocol_bytes=int(inbound_request_bytes + local_response_bytes),
            total_remote_compute_time_sec=local_compute_time_sec,
            timestamp_completed_ns=time.time_ns(),
        )
        return terminal

    if runtime.next_worker_cfg is None:
        raise RuntimeError(
            f"Worker {runtime.worker_id} produced non-terminal status "
            f"'{output.status}' but no next worker is configured"
        )

    activation = _extract_activation_cpu(output.activation)
    activation_bytes, activation_shape, activation_dtype = tensor_to_bytes(activation)

    next_metadata = InferenceRequestMetadata(
        request_id=metadata.request_id,
        sample_id=metadata.sample_id,
        trace_id=metadata.trace_id,
        request_kind=REQUEST_KIND_ACTIVATION,
        stage_id=runtime.partition_id + 1,
        origin_node=metadata.origin_node,
        current_node=str(runtime.next_worker_cfg["worker_id"]),
        next_node=runtime.next_worker_cfg.get("next_worker_id"),
        tensor_shape=activation_shape,
        tensor_dtype=activation_dtype,
        tensor_layout=metadata.tensor_layout,
        model_name=metadata.model_name,
        exit_policy=metadata.exit_policy,
        timestamp_sent_ns=time.time_ns(),
    )

    downstream_terminal, outbound_request_bytes, _ = infer_remote(
        worker_cfg=runtime.next_worker_cfg,
        metadata=next_metadata,
        tensor_bytes=activation_bytes,
    )

    response_bytes_from_this_stage = int(outbound_request_bytes)

    local_stage_metric = StageMetric(
        worker_id=runtime.worker_id,
        stage_id=runtime.partition_id,
        compute_time_sec=local_compute_time_sec,
        request_bytes=int(inbound_request_bytes),
        response_bytes=response_bytes_from_this_stage,
    )

    stage_metrics = [local_stage_metric, *downstream_terminal.stage_metrics]
    path = [runtime.worker_id, *downstream_terminal.path]

    total_request_bytes = int(inbound_request_bytes) + int(
        downstream_terminal.total_request_bytes
    )
    total_response_bytes = response_bytes_from_this_stage + int(
        downstream_terminal.total_response_bytes
    )
    total_protocol_bytes = total_request_bytes + total_response_bytes
    total_remote_compute_time_sec = (
        local_compute_time_sec + float(downstream_terminal.total_remote_compute_time_sec)
    )

    enriched_terminal = TerminalInferenceResponse(
        status=downstream_terminal.status,
        request_id=downstream_terminal.request_id,
        sample_id=downstream_terminal.sample_id,
        trace_id=downstream_terminal.trace_id,
        worker_id=downstream_terminal.worker_id,
        stage_id=downstream_terminal.stage_id,
        exit_id=downstream_terminal.exit_id,
        predicted_class=downstream_terminal.predicted_class,
        confidence=downstream_terminal.confidence,
        logits_shape=downstream_terminal.logits_shape,
        logits_dtype=downstream_terminal.logits_dtype,
        compute_time_sec=downstream_terminal.compute_time_sec,
        stage_metrics=stage_metrics,
        path=path,
        total_request_bytes=total_request_bytes,
        total_response_bytes=total_response_bytes,
        total_protocol_bytes=total_protocol_bytes,
        total_remote_compute_time_sec=total_remote_compute_time_sec,
        timestamp_completed_ns=downstream_terminal.timestamp_completed_ns,
    )
    return enriched_terminal


def _extract_logits_cpu(logits: torch.Tensor | None) -> torch.Tensor:
    if logits is None:
        raise RuntimeError("Expected terminal output logits, got None")
    return logits.detach().cpu().contiguous()


def _extract_activation_cpu(activation: torch.Tensor | None) -> torch.Tensor:
    if activation is None:
        raise RuntimeError("Expected forwarded activation, got None")
    return activation.detach().cpu().contiguous()


def _compute_prediction_summary(logits: torch.Tensor) -> tuple[int, float]:
    if logits.ndim < 2:
        raise ValueError(f"Expected logits with ndim >= 2, got shape {list(logits.shape)}")

    probs = torch.softmax(logits, dim=1)
    confidence_tensor, predicted_tensor = probs.max(dim=1)

    predicted_class = int(predicted_tensor[0].item())
    confidence = float(confidence_tensor[0].item())
    return predicted_class, confidence


def _estimate_terminal_response_bytes(
    *,
    request_id: str,
    sample_id: int,
    trace_id: str,
    worker_id: str,
    stage_id: int,
    exit_id: int,
    predicted_class: int | None,
    confidence: float | None,
    logits_shape: list[int],
    logits_dtype: str,
) -> int:
    """
    Rough terminal JSON response estimate.

    This is intentionally approximate, but stable and comparable across runs.
    """
    payload = {
        "request_id": request_id,
        "sample_id": sample_id,
        "trace_id": trace_id,
        "worker_id": worker_id,
        "stage_id": stage_id,
        "exit_id": exit_id,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "logits_shape": logits_shape,
        "logits_dtype": logits_dtype,
    }

    body_estimate = len(str(payload).encode("utf-8"))
    json_overhead = 512
    http_overhead = 512
    return body_estimate + json_overhead + http_overhead