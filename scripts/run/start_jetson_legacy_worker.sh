#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <experiment_config> <worker_id>"
  exit 1
fi

CONFIG_PATH="$1"
WORKER_ID="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

echo "[start_jetson_legacy_worker] repo_root=$REPO_ROOT"
echo "[start_jetson_legacy_worker] config=$CONFIG_PATH"
echo "[start_jetson_legacy_worker] worker_id=$WORKER_ID"
echo "[start_jetson_legacy_worker] python=$PYTHON_BIN"

python3 -m src.distributed_legacy.server \
  --config "$CONFIG_PATH" \
  --worker-id "$WORKER_ID"
