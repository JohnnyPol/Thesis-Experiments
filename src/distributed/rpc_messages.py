from __future__ import annotations

from typing import Any


REQUEST_TYPE_INFER_INPUT = "infer_from_input"
REQUEST_TYPE_INFER_ACTIVATION = "infer_from_activation"

RESPONSE_STATUS_FORWARDED = "forward"
RESPONSE_STATUS_EXITED = "exited"
RESPONSE_STATUS_COMPLETED = "completed"
RESPONSE_STATUS_ERROR = "error"


def make_infer_request(
    request_id: str,
    sample_id: int,
    stage_id: int,
    tensor_bytes: bytes,
    request_type: str,
) -> dict[str, Any]:
    return {
        "message_type": "inference_request",
        "request_id": request_id,
        "sample_id": int(sample_id),
        "stage_id": int(stage_id),
        "request_type": request_type,
        "tensor_bytes": tensor_bytes,
    }


def make_forward_response(
    request_id: str,
    sample_id: int,
    stage_id: int,
    activation_bytes: bytes,
    compute_time_sec: float,
) -> dict[str, Any]:
    return {
        "message_type": "inference_response",
        "status": RESPONSE_STATUS_FORWARDED,
        "request_id": request_id,
        "sample_id": int(sample_id),
        "stage_id": int(stage_id),
        "exit_id": None,
        "logits_bytes": None,
        "activation_bytes": activation_bytes,
        "compute_time_sec": float(compute_time_sec),
    }


def make_terminal_response(
    request_id: str,
    sample_id: int,
    stage_id: int,
    status: str,
    exit_id: int,
    logits_bytes: bytes,
    compute_time_sec: float,
) -> dict[str, Any]:
    if status not in {RESPONSE_STATUS_EXITED, RESPONSE_STATUS_COMPLETED}:
        raise ValueError(f"Invalid terminal status: {status}")

    return {
        "message_type": "inference_response",
        "status": status,
        "request_id": request_id,
        "sample_id": int(sample_id),
        "stage_id": int(stage_id),
        "exit_id": int(exit_id),
        "logits_bytes": logits_bytes,
        "activation_bytes": None,
        "compute_time_sec": float(compute_time_sec),
    }


def make_error_response(
    request_id: str,
    sample_id: int,
    stage_id: int,
    error_message: str,
) -> dict[str, Any]:
    return {
        "message_type": "inference_response",
        "status": RESPONSE_STATUS_ERROR,
        "request_id": request_id,
        "sample_id": int(sample_id),
        "stage_id": int(stage_id),
        "exit_id": None,
        "logits_bytes": None,
        "activation_bytes": None,
        "compute_time_sec": 0.0,
        "error_message": str(error_message),
    }


def validate_request(message: dict[str, Any]) -> None:
    if message.get("message_type") != "inference_request":
        raise ValueError("Invalid request: message_type must be 'inference_request'")

    if message.get("request_type") not in {
        REQUEST_TYPE_INFER_INPUT,
        REQUEST_TYPE_INFER_ACTIVATION,
    }:
        raise ValueError("Invalid request_type")

    if "tensor_bytes" not in message or message["tensor_bytes"] is None:
        raise ValueError("Request must contain tensor_bytes")


def validate_response(message: dict[str, Any]) -> None:
    if message.get("message_type") != "inference_response":
        raise ValueError("Invalid response: message_type must be 'inference_response'")

    status = message.get("status")
    if status not in {
        RESPONSE_STATUS_FORWARDED,
        RESPONSE_STATUS_EXITED,
        RESPONSE_STATUS_COMPLETED,
        RESPONSE_STATUS_ERROR,
    }:
        raise ValueError(f"Invalid response status: {status}")