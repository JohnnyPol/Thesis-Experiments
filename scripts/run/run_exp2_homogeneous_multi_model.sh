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
    "configs/experiments/exp2_resnet18_cifar10.yaml"
    "configs/experiments/exp2_resnet18_cifar100.yaml"
    "configs/experiments/exp2_resnet34_cifar10.yaml"
    "configs/experiments/exp2_resnet34_cifar100.yaml"
  )
fi

echo "[run_exp2] repo_root=$REPO_ROOT"

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  echo "[run_exp2] config=$CONFIG_PATH"
  check_workers_for_config "[run_exp2]" "$CONFIG_PATH"

  python -m src.distributed.multi_model_master_client \
    --config "$CONFIG_PATH"
done
