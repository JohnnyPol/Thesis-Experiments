#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

source "${REPO_ROOT}/scripts/run/distributed_common.sh"

if [[ $# -gt 0 ]]; then
  CONFIG_PATHS=("$@")
else
  CONFIG_PATHS=(
    "configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10_1.yaml"
    "configs/experiments/exp1_3_resnet34_ee_3nodes_cifar10_1.yaml"
  )
fi

echo "[run_exp1_3_cifar10_1] repo_root=$REPO_ROOT"

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  echo "[run_exp1_3_cifar10_1] config=$CONFIG_PATH"
  check_workers_for_config "[run_exp1_3_cifar10_1]" "$CONFIG_PATH"

  python -m src.distributed.master_client \
    --config "$CONFIG_PATH"
done
