from __future__ import annotations

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
) -> TerminalInferenceResponse:
    """
    Send an inference request to a worker FastAPI endpoint.
    """
    url = f"{_worker_base_url(worker_cfg)}/infer"

    metadata_json = metadata.model_dump_json()
    files = {
        METADATA_FORM_FIELD: (None, metadata_json, "application/json"),
        TENSOR_FORM_FIELD: ("tensor.bin", tensor_bytes, "application/octet-stream"),
    }

    response = requests.post(url, files=files, timeout=timeout_sec)

    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == RESPONSE_STATUS_ERROR:
        error = ErrorResponse.model_validate(payload)
        raise RuntimeError(
            f"Remote worker error from {worker_cfg.get('worker_id', 'unknown')}: "
            f"{error.error_message}"
        )

    terminal = TerminalInferenceResponse.model_validate(payload)
    return terminal


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


def start_monitoring(
    worker_cfg: dict[str, Any],
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    url = f"{_worker_base_url(worker_cfg)}/monitoring/start"
    response = requests.post(url, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def stop_monitoring(
    worker_cfg: dict[str, Any],
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    url = f"{_worker_base_url(worker_cfg)}/monitoring/stop"
    response = requests.post(url, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


