#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

CONFIG_PATHS=(
  "${PROJECT_ROOT}/configs/experiments/exp1_2_resnet18_ee_single_node_cifar10.yaml"
  "${PROJECT_ROOT}/configs/experiments/exp1_2_resnet34_ee_single_node_cifar10.yaml"
)

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  echo "[run_exp1_2_cifar10] config=$CONFIG_PATH"
  python -m src.inference.single_node \
    --config "$CONFIG_PATH"
done
