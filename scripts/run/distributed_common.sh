#!/usr/bin/env bash

load_worker_expectations() {
  local config_path="$1"

  python - "$config_path" <<'PY'
import sys
import yaml
from pathlib import Path

exp_cfg_path = Path(sys.argv[1])

with open(exp_cfg_path, "r", encoding="utf-8") as handle:
    exp_cfg = yaml.safe_load(handle)

repo_root = Path.cwd()
system_cfg_path = (repo_root / exp_cfg["config_refs"]["system"]).resolve()
model_cfg_path = (repo_root / exp_cfg["config_refs"]["model"]).resolve()
dataset_cfg_path = (repo_root / exp_cfg["config_refs"]["dataset"]).resolve()

with open(system_cfg_path, "r", encoding="utf-8") as handle:
    system_cfg = yaml.safe_load(handle)
with open(model_cfg_path, "r", encoding="utf-8") as handle:
    model_cfg = yaml.safe_load(handle)
with open(dataset_cfg_path, "r", encoding="utf-8") as handle:
    dataset_cfg = yaml.safe_load(handle)

workers_by_id = {
    str(worker["worker_id"]): worker
    for worker in system_cfg.get("workers", [])
}

pipeline_order = system_cfg.get("pipeline_order", [])
if pipeline_order:
    ordered_workers = [workers_by_id[str(worker_id)] for worker_id in pipeline_order]
else:
    ordered_workers = sorted(
        system_cfg.get("workers", []),
        key=lambda worker: int(worker.get("partition_id", 0)),
    )

expected_model = str(model_cfg.get("name", ""))
expected_dataset = str(dataset_cfg.get("name", ""))
for worker in ordered_workers:
    worker_id = str(worker["worker_id"])
    host = str(worker.get("connect_host", worker["host"]))
    port = int(worker["port"])
    partition_id = int(worker.get("partition_id", 0))
    print(f"{worker_id} {host} {port} {partition_id} {expected_model} {expected_dataset}")
PY
}

validate_worker_info() {
  local prefix="$1"
  local config_path="$2"
  local worker_id="$3"
  local expected_partition_id="$4"
  local expected_model="$5"
  local expected_dataset="$6"
  local info_json="$7"

  INFO_JSON="$info_json" python - \
    "$prefix" "$config_path" "$worker_id" "$expected_partition_id" \
    "$expected_model" "$expected_dataset" <<'PY'
import json
import os
import sys

prefix, config_path, worker_id, expected_partition_id, expected_model, expected_dataset = sys.argv[1:]
info = json.loads(os.environ["INFO_JSON"])

actual_worker = str(info.get("worker_id"))
actual_partition = str(info.get("partition_id"))
actual_model = str(info.get("model_name"))
actual_dataset = str(info.get("dataset_name"))

errors = []
if actual_worker != worker_id:
    errors.append(f"worker_id={actual_worker!r}, expected {worker_id!r}")
if actual_partition != str(expected_partition_id):
    errors.append(
        f"partition_id={actual_partition!r}, expected {str(expected_partition_id)!r}"
    )
if actual_model != expected_model:
    errors.append(f"model_name={actual_model!r}, expected {expected_model!r}")
if actual_dataset != expected_dataset:
    errors.append(f"dataset_name={actual_dataset!r}, expected {expected_dataset!r}")

if errors:
    print(f"{prefix} {worker_id} is running the wrong config:", file=sys.stderr)
    for error in errors:
        print(f"{prefix}   - {error}", file=sys.stderr)
    print(
        f"{prefix} restart it with: bash scripts/run/start_worker_api.sh {config_path} {worker_id}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

check_workers_for_config() {
  local prefix="$1"
  local config_path="$2"

  mapfile -t worker_expectations < <(load_worker_expectations "$config_path")

  if [[ ${#worker_expectations[@]} -eq 0 ]]; then
    echo "$prefix no workers found in system config"
    exit 1
  fi

  for entry in "${worker_expectations[@]}"; do
    read -r worker_id worker_host worker_port partition_id expected_model expected_dataset <<< "$entry"
    echo "$prefix checking ${worker_id} health at ${worker_host}:${worker_port}..."
    curl --fail --silent "http://${worker_host}:${worker_port}/health" >/dev/null
    info_json="$(curl --fail --silent "http://${worker_host}:${worker_port}/info")"
    validate_worker_info \
      "$prefix" \
      "$config_path" \
      "$worker_id" \
      "$partition_id" \
      "$expected_model" \
      "$expected_dataset" \
      "$info_json"
    echo "$prefix ${worker_id} is reachable and running ${expected_model}/${expected_dataset}"
  done
}
