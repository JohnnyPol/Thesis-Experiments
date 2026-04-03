from __future__ import annotations

import json
from typing import Any

import requests

from src.distributed.api.schemas import (
    ErrorResponse,
    InferenceRequestMetadata,
    TerminalInferenceResponse,
)
from src.distributed.protocol.constants import (
    DEFAULT_TIMEOUT_SEC,
    METADATA_FORM_FIELD,
    RESPONSE_STATUS_ERROR,
    TENSOR_FORM_FIELD,
)


def _worker_base_url(worker_cfg: dict[str, Any]) -> str:
    host = str(worker_cfg.get("connect_host", worker_cfg["host"]))
    port = int(worker_cfg["port"])
    return f"http://{host}:{port}"


def infer_remote(
    worker_cfg: dict[str, Any],
    metadata: InferenceRequestMetadata,
    tensor_bytes: bytes,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> tuple[TerminalInferenceResponse, int, int]:
    """
    Send an inference request to a worker FastAPI endpoint.

    Returns:
        (terminal_response, estimated_request_bytes, estimated_response_bytes)
    """
    url = f"{_worker_base_url(worker_cfg)}/infer"

    metadata_json = metadata.model_dump_json()
    files = {
        METADATA_FORM_FIELD: (None, metadata_json, "application/json"),
        TENSOR_FORM_FIELD: ("tensor.bin", tensor_bytes, "application/octet-stream"),
    }

    response = requests.post(url, files=files, timeout=timeout_sec)

    estimated_request_bytes = _estimate_request_bytes(
        url=url,
        metadata_json=metadata_json,
        tensor_bytes=tensor_bytes,
    )
    estimated_response_bytes = _estimate_response_bytes(response)

    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == RESPONSE_STATUS_ERROR:
        error = ErrorResponse.model_validate(payload)
        raise RuntimeError(
            f"Remote worker error from {worker_cfg.get('worker_id', 'unknown')}: "
            f"{error.error_message}"
        )

    terminal = TerminalInferenceResponse.model_validate(payload)
    return terminal, estimated_request_bytes, estimated_response_bytes


def get_health(
    worker_cfg: dict[str, Any],
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    url = f"{_worker_base_url(worker_cfg)}/health"
    response = requests.get(url, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def get_info(
    worker_cfg: dict[str, Any],
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    url = f"{_worker_base_url(worker_cfg)}/info"
    response = requests.get(url, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def _estimate_request_bytes(
    url: str,
    metadata_json: str,
    tensor_bytes: bytes,
) -> int:
    """
    Rough request size estimate for metrics purposes.

    This is not exact wire-level TCP accounting, but it is stable enough for
    research-side protocol byte comparisons.
    """
    url_bytes = len(url.encode("utf-8"))
    metadata_bytes = len(metadata_json.encode("utf-8"))
    tensor_nbytes = len(tensor_bytes)

    multipart_overhead = 512
    http_header_overhead = 512

    return url_bytes + metadata_bytes + tensor_nbytes + multipart_overhead + http_header_overhead


def _estimate_response_bytes(response: requests.Response) -> int:
    """
    Rough response size estimate based on response body + header approximation.
    """
    body_bytes = len(response.content)
    header_bytes = sum(
        len(str(key).encode("utf-8")) + len(str(value).encode("utf-8"))
        for key, value in response.headers.items()
    )
    http_status_line_overhead = 64
    return body_bytes + header_bytes + http_status_line_overhead