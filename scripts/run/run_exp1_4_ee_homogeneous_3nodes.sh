#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp1_4_ee_homogeneous_3nodes.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

echo "[run_exp1_4] repo_root=$REPO_ROOT"
echo "[run_exp1_4] config=$CONFIG_PATH"

mapfile -t WORKER_ENDPOINTS < <(
python - <<'PY' "$CONFIG_PATH"
import sys
import yaml
from pathlib import Path

exp_cfg_path = Path(sys.argv[1])

with open(exp_cfg_path, "r", encoding="utf-8") as f:
    exp_cfg = yaml.safe_load(f)

system_ref = exp_cfg["config_refs"]["system"]
repo_root = Path.cwd()
system_cfg_path = (repo_root / system_ref).resolve()

with open(system_cfg_path, "r", encoding="utf-8") as f:
    system_cfg = yaml.safe_load(f)

workers_by_id = {
    str(worker["worker_id"]): worker
    for worker in system_cfg.get("workers", [])
}

pipeline_order = system_cfg.get("pipeline_order", [])
if not pipeline_order:
    ordered_workers = sorted(
        system_cfg.get("workers", []),
        key=lambda w: int(w.get("partition_id", 0)),
    )
else:
    ordered_workers = [workers_by_id[str(worker_id)] for worker_id in pipeline_order]

for worker in ordered_workers:
    worker_id = str(worker["worker_id"])
    host = str(worker.get("connect_host", worker["host"]))
    port = int(worker["port"])
    print(f"{worker_id} {host} {port}")
PY
)

if [[ ${#WORKER_ENDPOINTS[@]} -eq 0 ]]; then
  echo "[run_exp1_4] no workers found in system config"
  exit 1
fi

for entry in "${WORKER_ENDPOINTS[@]}"; do
  read -r WORKER_ID WORKER_HOST WORKER_PORT <<< "$entry"
  echo "[run_exp1_4] checking ${WORKER_ID} health at ${WORKER_HOST}:${WORKER_PORT}..."
  curl --fail --silent "http://${WORKER_HOST}:${WORKER_PORT}/health" >/dev/null
  echo "[run_exp1_4] ${WORKER_ID} is reachable"
done

python -m src.distributed.master_client \
  --config "$CONFIG_PATH"