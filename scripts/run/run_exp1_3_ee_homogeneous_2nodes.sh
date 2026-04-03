#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp1_3_ee_homogeneous_2nodes.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

echo "[run_exp1_3] repo_root=$REPO_ROOT"
echo "[run_exp1_3] config=$CONFIG_PATH"

python -m src.distributed.master_client \
  --config "$CONFIG_PATH"