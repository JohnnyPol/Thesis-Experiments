from __future__ import annotations

import argparse
import socket
import traceback
from typing import Any

import torch

from src.distributed.rpc_messages import (
    REQUEST_TYPE_INFER_ACTIVATION,
    REQUEST_TYPE_INFER_INPUT,
    RESPONSE_STATUS_COMPLETED,
    RESPONSE_STATUS_EXITED,
    RESPONSE_STATUS_FORWARDED,
    make_error_response,
    make_forward_response,
    make_terminal_response,
    validate_request,
)
from src.distributed.serialization import bytes_to_tensor, recv_message, send_message, tensor_to_bytes
from src.models.partitioning import build_partition_module
from src.utils.config import load_experiment_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed worker server for partitioned EE inference.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    parser.add_argument("--worker-id", type=str, required=True, help="Worker identifier from system config.")
    return parser.parse_args()


def find_worker_cfg(system_cfg: dict[str, Any], worker_id: str) -> dict[str, Any]:
    workers = system_cfg.get("workers", [])
    for worker_cfg in workers:
        if worker_cfg.get("worker_id") == worker_id:
            return worker_cfg
    raise ValueError(f"Worker '{worker_id}' not found in system config")


def handle_request(
    request: dict[str, Any],
    partition_module: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    validate_request(request)

    request_id = str(request["request_id"])
    sample_id = int(request["sample_id"])
    stage_id = int(request["stage_id"])
    request_type = str(request["request_type"])

    tensor = bytes_to_tensor(request["tensor_bytes"], device=device)
    if request_type not in {REQUEST_TYPE_INFER_INPUT, REQUEST_TYPE_INFER_ACTIVATION}:
        return make_error_response(
            request_id=request_id,
            sample_id=sample_id,
            stage_id=stage_id,
            error_message=f"Unsupported request_type: {request_type}",
        )

    with torch.no_grad():
        output = partition_module(tensor)

    if output.status == RESPONSE_STATUS_FORWARDED:
        assert output.activation is not None
        return make_forward_response(
            request_id=request_id,
            sample_id=sample_id,
            stage_id=stage_id,
            activation_bytes=tensor_to_bytes(output.activation),
            compute_time_sec=output.compute_time_sec,
        )

    if output.status in {RESPONSE_STATUS_EXITED, RESPONSE_STATUS_COMPLETED}:
        assert output.logits is not None
        assert output.exit_id is not None
        return make_terminal_response(
            request_id=request_id,
            sample_id=sample_id,
            stage_id=stage_id,
            status=output.status,
            exit_id=output.exit_id,
            logits_bytes=tensor_to_bytes(output.logits),
            compute_time_sec=output.compute_time_sec,
        )

    return make_error_response(
        request_id=request_id,
        sample_id=sample_id,
        stage_id=stage_id,
        error_message=f"Unexpected partition output status: {output.status}",
    )


def main() -> None:
    args = parse_args()

    bundle = load_experiment_bundle(args.config)
    dataset_cfg = bundle["dataset_config"]
    model_cfg = bundle["model_config"]
    system_cfg = bundle["system_config"]
    repo_root = bundle["repo_root"]

    worker_cfg = find_worker_cfg(system_cfg, args.worker_id)
    partition_id = int(worker_cfg["partition_id"])
    bind_host = str(worker_cfg.get("bind_host", worker_cfg["host"]))
    port = int(worker_cfg["port"])
    device = torch.device(worker_cfg.get("device", "cpu"))

    partition_module = build_partition_module(
        partition_id=partition_id,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        repo_root=repo_root,
        device=device,
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_host, port))
    server.listen(16)

    print(
        f"[worker_server] worker_id={args.worker_id} "
        f"partition_id={partition_id} "
        f"bind={bind_host}:{port} "
        f"device={device}"
    )

    try:
        while True:
            conn, addr = server.accept()
            with conn:
                try:
                    request, _ = recv_message(conn)
                    response = handle_request(
                        request=request,
                        partition_module=partition_module,
                        device=device,
                    )
                except Exception as exc:
                    request_id = request.get("request_id", "unknown") if "request" in locals() else "unknown"
                    sample_id = int(request.get("sample_id", -1)) if "request" in locals() else -1
                    stage_id = int(request.get("stage_id", -1)) if "request" in locals() else -1

                    traceback.print_exc()
                    response = make_error_response(
                        request_id=request_id,
                        sample_id=sample_id,
                        stage_id=stage_id,
                        error_message=str(exc),
                    )

                send_message(conn, response)
                print(f"[worker_server] handled request from {addr} status={response.get('status')}")
    finally:
        server.close()


if __name__ == "__main__":
    main()