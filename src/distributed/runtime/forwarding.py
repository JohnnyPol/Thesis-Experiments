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
    model_instance_id = str(metadata.model_instance_id or "model_0")
    current_metadata = metadata
    current_tensor = tensor
    current_inbound_request_bytes = int(inbound_request_bytes)

    local_stage_metrics: list[StageMetric] = []
    local_path: list[str] = []
    local_request_bytes_total = 0
    local_response_bytes_total = 0
    local_compute_time_total = 0.0

    while True:
        current_partition_id = runtime.resolve_current_partition_id(
            int(current_metadata.stage_id)
        )
        partition_module = runtime.get_partition_module(
            model_instance_id,
            current_partition_id,
        )
        tensor_on_device = current_tensor.to(runtime.device)

        with torch.no_grad():
            output = partition_module(tensor_on_device)

        local_compute_time_sec = float(output.compute_time_sec)
        local_compute_time_total += local_compute_time_sec
        local_request_bytes_total += int(current_inbound_request_bytes)

        if output.status in {RESPONSE_STATUS_EXITED, RESPONSE_STATUS_COMPLETED}:
            logits = _extract_logits_cpu(output.logits)
            predicted_class, confidence = _compute_prediction_summary(logits)

            logits_shape = list(logits.shape)
            logits_dtype = torch_dtype_to_str(logits.dtype)

            local_response_bytes = _estimate_terminal_response_bytes(
                request_id=current_metadata.request_id,
                sample_id=current_metadata.sample_id,
                trace_id=current_metadata.trace_id,
                worker_id=runtime.worker_id,
                stage_id=current_partition_id,
                exit_id=int(output.exit_id),
                predicted_class=predicted_class,
                confidence=confidence,
                logits_shape=logits_shape,
                logits_dtype=logits_dtype,
            )
            local_response_bytes_total += int(local_response_bytes)

            local_stage_metrics.append(
                StageMetric(
                    worker_id=runtime.worker_id,
                    stage_id=current_partition_id,
                    model_instance_id=model_instance_id,
                    compute_time_sec=local_compute_time_sec,
                    request_bytes=int(current_inbound_request_bytes),
                    response_bytes=int(local_response_bytes),
                )
            )
            local_path.append(runtime.worker_id)

            return TerminalInferenceResponse(
                status=output.status,
                request_id=current_metadata.request_id,
                sample_id=current_metadata.sample_id,
                trace_id=current_metadata.trace_id,
                model_instance_id=model_instance_id,
                worker_id=runtime.worker_id,
                stage_id=current_partition_id,
                exit_id=int(output.exit_id),
                predicted_class=predicted_class,
                confidence=confidence,
                logits_shape=logits_shape,
                logits_dtype=logits_dtype,
                compute_time_sec=local_compute_time_sec,
                stage_metrics=local_stage_metrics,
                path=local_path,
                total_request_bytes=int(local_request_bytes_total),
                total_response_bytes=int(local_response_bytes_total),
                total_protocol_bytes=int(
                    local_request_bytes_total + local_response_bytes_total
                ),
                total_remote_compute_time_sec=local_compute_time_total,
                timestamp_completed_ns=time.time_ns(),
            )

        next_route_entry = runtime.resolve_next_route_entry(
            model_instance_id,
            current_partition_id,
        )
        if next_route_entry is None:
            raise RuntimeError(
                f"Worker {runtime.worker_id} produced non-terminal status "
                f"'{output.status}' at stage {current_partition_id} but no next "
                "worker is configured"
            )

        activation = _extract_activation_cpu(output.activation)
        activation_bytes, activation_shape, activation_dtype = tensor_to_bytes(activation)

        next_worker_cfg = runtime.get_worker_cfg(next_route_entry.worker_id)
        next_after_next = runtime.peek_next_route_entry(
            model_instance_id,
            int(next_route_entry.partition_id),
        )

        next_metadata = InferenceRequestMetadata(
            request_id=current_metadata.request_id,
            sample_id=current_metadata.sample_id,
            trace_id=current_metadata.trace_id,
            model_instance_id=model_instance_id,
            request_kind=REQUEST_KIND_ACTIVATION,
            stage_id=int(next_route_entry.partition_id),
            origin_node=current_metadata.origin_node,
            current_node=str(next_route_entry.worker_id),
            next_node=(
                str(next_after_next.worker_id) if next_after_next is not None else None
            ),
            tensor_shape=activation_shape,
            tensor_dtype=activation_dtype,
            tensor_layout=current_metadata.tensor_layout,
            model_name=current_metadata.model_name,
            exit_policy=current_metadata.exit_policy,
            timestamp_sent_ns=time.time_ns(),
        )

        if str(next_route_entry.worker_id) == runtime.worker_id:
            local_stage_metrics.append(
                StageMetric(
                    worker_id=runtime.worker_id,
                    stage_id=current_partition_id,
                    model_instance_id=model_instance_id,
                    compute_time_sec=local_compute_time_sec,
                    request_bytes=int(current_inbound_request_bytes),
                    response_bytes=0,
                )
            )
            local_path.append(runtime.worker_id)
            current_metadata = next_metadata
            current_tensor = activation
            current_inbound_request_bytes = 0
            continue

        downstream_terminal, outbound_request_bytes, _ = infer_remote(
            worker_cfg=next_worker_cfg,
            metadata=next_metadata,
            tensor_bytes=activation_bytes,
        )

        response_bytes_from_this_stage = int(outbound_request_bytes)
        local_response_bytes_total += response_bytes_from_this_stage

        local_stage_metrics.append(
            StageMetric(
                worker_id=runtime.worker_id,
                stage_id=current_partition_id,
                model_instance_id=model_instance_id,
                compute_time_sec=local_compute_time_sec,
                request_bytes=int(current_inbound_request_bytes),
                response_bytes=response_bytes_from_this_stage,
            )
        )
        local_path.append(runtime.worker_id)

        stage_metrics = [*local_stage_metrics, *downstream_terminal.stage_metrics]
        path = [*local_path, *downstream_terminal.path]

        total_request_bytes = int(local_request_bytes_total) + int(
            downstream_terminal.total_request_bytes
        )
        total_response_bytes = int(local_response_bytes_total) + int(
            downstream_terminal.total_response_bytes
        )
        total_protocol_bytes = total_request_bytes + total_response_bytes
        total_remote_compute_time_sec = local_compute_time_total + float(
            downstream_terminal.total_remote_compute_time_sec
        )

        return TerminalInferenceResponse(
            status=downstream_terminal.status,
            request_id=downstream_terminal.request_id,
            sample_id=downstream_terminal.sample_id,
            trace_id=downstream_terminal.trace_id,
            model_instance_id=downstream_terminal.model_instance_id,
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
