from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.distributed.protocol.constants import (
    PROTOCOL_VERSION,
    REQUEST_KIND_ACTIVATION,
    REQUEST_KIND_INPUT,
    RESPONSE_STATUS_COMPLETED,
    RESPONSE_STATUS_ERROR,
    RESPONSE_STATUS_EXITED,
    TENSOR_LAYOUT_NCHW,
    VALID_REQUEST_KINDS,
    VALID_RESPONSE_STATUSES,
    VALID_TENSOR_LAYOUTS,
)


class StageMetric(BaseModel):
    worker_id: str = Field(..., description="Worker that executed this stage")
    stage_id: int = Field(..., ge=0, description="Logical stage/partition index")
    compute_time_sec: float = Field(..., ge=0.0)
    request_bytes: int = Field(..., ge=0)
    response_bytes: int = Field(..., ge=0)


class InferenceRequestMetadata(BaseModel):
    protocol_version: str = Field(default=PROTOCOL_VERSION)

    request_id: str
    sample_id: int = Field(..., ge=0)
    trace_id: str

    request_kind: str = Field(
        ...,
        description="Whether this request contains raw model input or a forwarded activation",
    )

    stage_id: int = Field(..., ge=0)
    origin_node: str = Field(..., description="Usually 'master' for the first request")
    current_node: str = Field(..., description="Node expected to handle this request")
    next_node: str | None = Field(
        default=None,
        description="Optional next-hop worker id if this stage forwards onward",
    )

    tensor_shape: list[int] = Field(..., min_length=1)
    tensor_dtype: str = Field(..., description="Transport dtype string, e.g. float32")
    tensor_layout: str = Field(default=TENSOR_LAYOUT_NCHW)

    model_name: str | None = None
    exit_policy: str | None = None

    timestamp_sent_ns: int | None = Field(default=None, ge=0)

    @field_validator("request_kind")
    @classmethod
    def validate_request_kind(cls, value: str) -> str:
        if value not in VALID_REQUEST_KINDS:
            raise ValueError(
                f"Invalid request_kind '{value}'. Expected one of {sorted(VALID_REQUEST_KINDS)}"
            )
        return value

    @field_validator("tensor_layout")
    @classmethod
    def validate_tensor_layout(cls, value: str) -> str:
        if value not in VALID_TENSOR_LAYOUTS:
            raise ValueError(
                f"Invalid tensor_layout '{value}'. Expected one of {sorted(VALID_TENSOR_LAYOUTS)}"
            )
        return value

    @field_validator("tensor_shape")
    @classmethod
    def validate_tensor_shape(cls, value: list[int]) -> list[int]:
        if len(value) == 0:
            raise ValueError("tensor_shape must not be empty")
        if any(dim <= 0 for dim in value):
            raise ValueError("tensor_shape must contain only positive dimensions")
        return value


class TerminalInferenceResponse(BaseModel):
    protocol_version: str = Field(default=PROTOCOL_VERSION)

    status: Literal["exited", "completed"]

    request_id: str
    sample_id: int = Field(..., ge=0)
    trace_id: str

    worker_id: str
    stage_id: int = Field(..., ge=0)
    exit_id: int = Field(..., ge=0)

    predicted_class: int | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    logits_shape: list[int] = Field(..., min_length=1)
    logits_dtype: str

    compute_time_sec: float = Field(
        ...,
        ge=0.0,
        description="Compute time for the worker returning this terminal response",
    )

    stage_metrics: list[StageMetric] = Field(default_factory=list)
    path: list[str] = Field(default_factory=list)

    total_request_bytes: int = Field(default=0, ge=0)
    total_response_bytes: int = Field(default=0, ge=0)
    total_protocol_bytes: int = Field(default=0, ge=0)
    total_remote_compute_time_sec: float = Field(default=0.0, ge=0.0)

    timestamp_completed_ns: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("path entries must be non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_totals(self) -> "TerminalInferenceResponse":
        if self.total_protocol_bytes != (
            self.total_request_bytes + self.total_response_bytes
        ):
            raise ValueError(
                "total_protocol_bytes must equal total_request_bytes + total_response_bytes"
            )

        stage_compute_sum = sum(metric.compute_time_sec for metric in self.stage_metrics)
        if self.total_remote_compute_time_sec + 1e-12 < stage_compute_sum:
            raise ValueError(
                "total_remote_compute_time_sec cannot be smaller than the sum of stage compute times"
            )

        return self


class ErrorResponse(BaseModel):
    protocol_version: str = Field(default=PROTOCOL_VERSION)

    status: Literal["error"]

    request_id: str
    sample_id: int = Field(default=-1)
    trace_id: str | None = None

    worker_id: str | None = None
    stage_id: int = Field(default=-1)

    error_message: str
    error_type: str | None = None


class WorkerInfoResponse(BaseModel):
    protocol_version: str = Field(default=PROTOCOL_VERSION)

    worker_id: str
    partition_id: int = Field(..., ge=0)
    num_partitions: int = Field(..., ge=1)

    host: str
    port: int = Field(..., ge=1, le=65535)
    device: str

    next_worker_id: str | None = None

    model_name: str | None = None
    exit_policy: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    worker_id: str