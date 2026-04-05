import argparse
import cgi
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from src.distributed_legacy.config import load_experiment_bundle
from src.distributed_legacy.protocol import (
    METADATA_FORM_FIELD,
    PROTOCOL_VERSION,
    RESPONSE_STATUS_COMPLETED,
    RESPONSE_STATUS_ERROR,
    RESPONSE_STATUS_EXITED,
    TENSOR_FORM_FIELD,
)
from src.distributed_legacy.runtime import build_worker_runtime
from src.distributed_legacy.tensor_codec import bytes_to_tensor, torch_dtype_to_str


class LegacyJetsonWorkerHandler(BaseHTTPRequestHandler):
    runtime = None

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "worker_id": self.runtime.worker_id,
                },
            )
            return

        if self.path == "/info":
            self._send_json(
                200,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": self.runtime.worker_id,
                    "partition_id": self.runtime.partition_id,
                    "num_partitions": self.runtime.num_partitions,
                    "host": self.runtime.host,
                    "port": self.runtime.port,
                    "device": str(self.runtime.device),
                    "next_worker_id": self.runtime.next_worker_id,
                    "model_name": self.runtime.model_name,
                    "exit_policy": self.runtime.exit_policy,
                },
            )
            return

        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path == "/monitoring/start":
            self.runtime.emissions_monitor.start()
            self._send_json(
                200,
                {
                    "status": "started",
                    "worker_id": self.runtime.worker_id,
                    "tracker_active": self.runtime.emissions_monitor.is_active(),
                },
            )
            return

        if self.path == "/monitoring/stop":
            carbon_kg, energy_kwh = self.runtime.emissions_monitor.stop()
            self._send_json(
                200,
                {
                    "status": "stopped",
                    "worker_id": self.runtime.worker_id,
                    "tracker_active": self.runtime.emissions_monitor.is_active(),
                    "carbon_kg": float(carbon_kg) if carbon_kg is not None else None,
                    "energy_kWh": float(energy_kwh) if energy_kwh is not None else None,
                },
            )
            return

        if self.path == "/infer":
            self._handle_infer()
            return

        self._send_json(404, {"error": "not_found"})

    def _handle_infer(self):
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                },
            )

            metadata_raw = form[METADATA_FORM_FIELD].value
            tensor_field = form[TENSOR_FORM_FIELD]
            raw_payload = tensor_field.file.read()

            metadata = json.loads(metadata_raw)
            tensor = bytes_to_tensor(
                payload=raw_payload,
                shape=metadata["tensor_shape"],
                dtype_str=metadata["tensor_dtype"],
                device="cpu",
            )

            inbound_request_bytes = self._estimate_inbound_request_bytes(
                metadata_raw,
                len(raw_payload),
            )

            terminal = execute_terminal_inference(
                runtime=self.runtime,
                metadata=metadata,
                tensor=tensor,
                inbound_request_bytes=inbound_request_bytes,
            )
            self._send_json(200, terminal)
        except Exception as exc:
            self._send_json(
                500,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": RESPONSE_STATUS_ERROR,
                    "request_id": "unknown",
                    "sample_id": -1,
                    "trace_id": None,
                    "worker_id": self.runtime.worker_id,
                    "stage_id": self.runtime.partition_id,
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    def _estimate_inbound_request_bytes(self, metadata_str, tensor_nbytes):
        header_bytes = 0
        for key, value in self.headers.items():
            header_bytes += len(str(key).encode("utf-8")) + len(
                str(value).encode("utf-8")
            )

        metadata_bytes = len(metadata_str.encode("utf-8"))
        multipart_overhead = 512
        http_overhead = 256

        return header_bytes + metadata_bytes + tensor_nbytes + multipart_overhead + http_overhead

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def execute_terminal_inference(runtime, metadata, tensor, inbound_request_bytes):
    tensor_on_device = tensor.to(runtime.device)

    with torch.no_grad():
        output = runtime.partition_module(tensor_on_device)

    local_compute_time_sec = float(output.compute_time_sec)
    if output.status not in (RESPONSE_STATUS_EXITED, RESPONSE_STATUS_COMPLETED):
        raise RuntimeError(
            "Legacy Jetson runtime expected a terminal partition, got status '{0}'".format(
                output.status
            )
        )

    logits = output.logits
    if logits is None:
        raise RuntimeError("Expected terminal output logits, got None")
    logits = logits.detach().cpu().contiguous()

    probs = torch.softmax(logits, dim=1)
    confidence_tensor, predicted_tensor = probs.max(dim=1)
    predicted_class = int(predicted_tensor[0].item())
    confidence = float(confidence_tensor[0].item())

    logits_shape = list(logits.shape)
    logits_dtype = torch_dtype_to_str(logits.dtype)
    response_bytes = estimate_terminal_response_bytes(
        metadata,
        runtime.worker_id,
        runtime.partition_id,
        int(output.exit_id),
        predicted_class,
        confidence,
        logits_shape,
        logits_dtype,
    )

    stage_metric = {
        "worker_id": runtime.worker_id,
        "stage_id": runtime.partition_id,
        "compute_time_sec": local_compute_time_sec,
        "request_bytes": int(inbound_request_bytes),
        "response_bytes": int(response_bytes),
    }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": output.status,
        "request_id": metadata["request_id"],
        "sample_id": int(metadata["sample_id"]),
        "trace_id": metadata["trace_id"],
        "worker_id": runtime.worker_id,
        "stage_id": runtime.partition_id,
        "exit_id": int(output.exit_id),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "logits_shape": logits_shape,
        "logits_dtype": logits_dtype,
        "compute_time_sec": local_compute_time_sec,
        "stage_metrics": [stage_metric],
        "path": [runtime.worker_id],
        "total_request_bytes": int(inbound_request_bytes),
        "total_response_bytes": int(response_bytes),
        "total_protocol_bytes": int(inbound_request_bytes + response_bytes),
        "total_remote_compute_time_sec": local_compute_time_sec,
        "timestamp_completed_ns": int(time.time() * 1000000000),
    }


def estimate_terminal_response_bytes(
    metadata,
    worker_id,
    stage_id,
    exit_id,
    predicted_class,
    confidence,
    logits_shape,
    logits_dtype,
):
    payload = {
        "request_id": metadata["request_id"],
        "sample_id": metadata["sample_id"],
        "trace_id": metadata["trace_id"],
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy Python 3.6 worker service for Jetson Nano final-stage inference."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_experiment_bundle(args.config)

    runtime = build_worker_runtime(
        worker_id=args.worker_id,
        dataset_cfg=bundle["dataset_config"],
        model_cfg=bundle["model_config"],
        system_cfg=bundle["system_config"],
        repo_root=bundle["repo_root"],
    )

    bind_host = args.host or str(runtime.worker_cfg.get("bind_host", "0.0.0.0"))
    port = args.port or int(runtime.port)

    LegacyJetsonWorkerHandler.runtime = runtime
    server = HTTPServer((bind_host, port), LegacyJetsonWorkerHandler)

    print(
        "Legacy Jetson worker listening on {0}:{1} for worker_id={2}".format(
            bind_host, port, runtime.worker_id
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
