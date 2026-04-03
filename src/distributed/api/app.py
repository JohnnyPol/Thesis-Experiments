from __future__ import annotations

import argparse

from fastapi import FastAPI
import uvicorn

from src.distributed.api.routes import create_router
from src.distributed.runtime.worker_runtime import build_worker_runtime
from src.utils.config import load_experiment_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastAPI worker service for distributed early-exit inference."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        required=True,
        help="Worker identifier from system config.",
    )
    return parser.parse_args()


def create_app_from_config(config_path: str, worker_id: str) -> FastAPI:
    bundle = load_experiment_bundle(config_path)

    dataset_cfg = bundle["dataset_config"]
    model_cfg = bundle["model_config"]
    system_cfg = bundle["system_config"]
    repo_root = bundle["repo_root"]

    runtime = build_worker_runtime(
        worker_id=worker_id,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        system_cfg=system_cfg,
        repo_root=repo_root,
    )

    app = FastAPI(
        title=f"Distributed EE Worker {runtime.worker_id}",
        version="1.0",
    )
    app.state.runtime = runtime
    app.include_router(create_router(runtime))

    return app


def main() -> None:
    args = parse_args()
    app = create_app_from_config(args.config, args.worker_id)

    runtime = app.state.runtime
    bind_host = str(runtime.worker_cfg.get("bind_host", "0.0.0.0"))
    port = int(runtime.port)

    uvicorn.run(
        app,
        host=bind_host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()