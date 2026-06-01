from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.distributed.api.schemas import (
    ErrorResponse,
    HealthResponse,
    InferenceRequestMetadata,
    TerminalInferenceResponse,
    WorkerMonitoringStartResponse,
    WorkerMonitoringStopResponse,
    WorkerInfoResponse,
)
from src.distributed.protocol.constants import METADATA_FORM_FIELD, TENSOR_FORM_FIELD
from src.distributed.protocol.tensor_codec import bytes_to_tensor
from src.distributed.runtime.forwarding import execute_or_forward
from src.distributed.runtime.worker_runtime import WorkerRuntime


def create_router(runtime: WorkerRuntime) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(worker_id=runtime.worker_id)

    @router.get("/info", response_model=WorkerInfoResponse)
    def info() -> WorkerInfoResponse:
        return WorkerInfoResponse(
            worker_id=runtime.worker_id,
            partition_id=runtime.partition_id,
            num_partitions=runtime.num_partitions,
            host=runtime.host,
            port=runtime.port,
            device=str(runtime.device),
            next_worker_id=runtime.next_worker_id,
            model_instance_ids=runtime.model_instance_ids,
            model_name=runtime.model_name,
            dataset_name=runtime.dataset_name,
            exit_policy=runtime.exit_policy,
        )

    @router.post("/monitoring/start", response_model=WorkerMonitoringStartResponse)
    def start_monitoring() -> WorkerMonitoringStartResponse:
        runtime.emissions_monitor.start()
        return WorkerMonitoringStartResponse(
            status="started",
            worker_id=runtime.worker_id,
            tracker_active=runtime.emissions_monitor.is_active,
        )

    @router.post("/monitoring/stop", response_model=WorkerMonitoringStopResponse)
    def stop_monitoring() -> WorkerMonitoringStopResponse:
        carbon_kg, energy_kwh = runtime.emissions_monitor.stop()
        return WorkerMonitoringStopResponse(
            status="stopped",
            worker_id=runtime.worker_id,
            tracker_active=runtime.emissions_monitor.is_active,
            carbon_kg=carbon_kg,
            energy_kWh=energy_kwh,
        )

    @router.post(
        "/infer",
        response_model=TerminalInferenceResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def infer(
        metadata: str = Form(..., alias=METADATA_FORM_FIELD),
        tensor_file: UploadFile = File(..., alias=TENSOR_FORM_FIELD),
    ) -> TerminalInferenceResponse | JSONResponse:
        raw_payload = await tensor_file.read()

        try:
            metadata_dict = json.loads(metadata)
            meta = InferenceRequestMetadata.model_validate(metadata_dict)

            tensor = bytes_to_tensor(
                payload=raw_payload,
                shape=meta.tensor_shape,
                dtype_str=meta.tensor_dtype,
                device="cpu",
            )

            terminal = await run_in_threadpool(
                execute_or_forward,
                runtime=runtime,
                metadata=meta,
                tensor=tensor,
            )
            return terminal

        except Exception as exc:
            error = ErrorResponse(
                status="error",
                request_id=_safe_request_id(metadata),
                sample_id=_safe_sample_id(metadata),
                trace_id=_safe_trace_id(metadata),
                worker_id=runtime.worker_id,
                stage_id=runtime.partition_id,
                error_message=str(exc),
                error_type=type(exc).__name__,
            )
            return JSONResponse(status_code=500, content=error.model_dump())

    return router


def _safe_request_id(metadata_str: str) -> str:
    try:
        return str(json.loads(metadata_str).get("request_id", "unknown"))
    except Exception:
        return "unknown"


def _safe_sample_id(metadata_str: str) -> int:
    try:
        return int(json.loads(metadata_str).get("sample_id", -1))
    except Exception:
        return -1


def _safe_trace_id(metadata_str: str) -> str | None:
    try:
        value = json.loads(metadata_str).get("trace_id")
        return None if value is None else str(value)
    except Exception:
        return None
